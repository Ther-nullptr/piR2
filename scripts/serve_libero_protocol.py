"""GPU backend with explicit request ticks and valid action timestamps."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from gr00t.policy import gr00t_policy as policy_module
from gr00t.policy.decoupled_policy import DecoupledGr00tPolicy
from gr00t.policy.server_client import PolicyServer
from libero_experiment_utils import checkpoint_identity
from libero_protocol_scheduler import extra_buffer_shift, project_commands
from transformers import BatchFeature

from coexecution.groot_optimization import (
    GrootOptimizations,
    add_optimization_arguments,
    optimization_config,
)

KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


class Backend:
    def __init__(self, core, variant, output):
        self.core = core
        self.variant = variant
        self.output = output
        self.anchor = None
        self.cache_meta = {}
        self.counts = {"vlm": 0, "dit": 0}
        for name, module in [
            ("vlm", core.model.backbone),
            ("dit", core.model.action_head.model),
        ]:

            def hook(_m, _i, _o, key=name):
                self.counts[key] += 1

            module.register_forward_hook(hook)

    @staticmethod
    def raw(actions):
        return project_commands(np.concatenate([actions[k][0] for k in KEYS], axis=-1))

    @staticmethod
    def frame_hashes(observation):
        return {
            k: hashlib.sha256(v.tobytes()).hexdigest()
            for k, v in observation.get("video", {}).items()
        }

    def reset(self, options=None):
        self.core.reset_streaming_buffer()
        with self.core._vl_cache_lock:
            self.core._vl_cache = None
            self.core._vl_cache_t_image_capture = None
        self.anchor = None
        self.cache_meta = {}
        torch.manual_seed(int((options or {}).get("seed", 1000)))
        return {"reset": True}

    def vision(
        self,
        observation,
        capture_s,
        source_tick,
        requested_tick=None,
        return_features=True,
    ):
        before = dict(self.counts)
        torch.cuda.synchronize()
        started = time.monotonic()
        result = self.core.update_vlm_cache(
            observation, return_features=return_features, t_image_capture=capture_s
        )
        torch.cuda.synchronize()
        meta = {
            "capture_s": capture_s,
            "source_tick": source_tick,
            "requested_tick": source_tick if requested_tick is None else requested_tick,
            "frame_hashes": self.frame_hashes(observation),
            "vlm_seconds": time.monotonic() - started,
            "vlm_forward_calls": self.counts["vlm"] - before["vlm"],
        }
        self.cache_meta = meta
        result["meta"] = meta
        return result

    def install(self, vl_embeds, meta):
        features = {}
        for k, v in vl_embeds.items():
            t = torch.from_numpy(v).to(self.core.model.device)
            if t.dtype.is_floating_point:
                t = t.to(self.core.model.dtype)
            features[k] = t
        torch.cuda.synchronize()
        cid, _ = self.core._prime_vlm_cache(BatchFeature(features), meta["capture_s"])
        self.cache_meta = dict(meta)
        return {"cache_id": cid}

    def bootstrap(self, observation, capture_s, delay_ticks, seed=1000):
        self.reset({"seed": seed})
        before = dict(self.counts)
        torch.cuda.synchronize()
        started = time.monotonic()
        if self.variant == "pir2":
            actions, _ = self.core.seed_streaming_from_obs(
                observation,
                num_inference_timesteps=4,
                t_image_capture=capture_s,
                slide_steps=delay_ticks,
            )
            self.anchor = 0
            self.cache_meta = {
                "capture_s": capture_s,
                "source_tick": 0,
                "requested_tick": 0,
                "frame_hashes": self.frame_hashes(observation),
            }
        else:
            self.vision(observation, capture_s, 0, return_features=False)
            actions, _ = self.core.get_action_chunk_cached(
                {"state": observation["state"]},
                options={"num_inference_timesteps": 4},
                t_state_capture=capture_s,
                period_ms=50.0,
            )
        torch.cuda.synchronize()
        return {
            "actions": self.raw(actions),
            "bootstrap_seconds": time.monotonic() - started,
            "forward_counts": {k: self.counts[k] - before[k] for k in before},
        }

    def plan(
        self,
        observation,
        request_tick,
        delay_ticks,
        committed_actions,
        state_capture_s,
        visual=None,
        output_scope="native",
    ):
        r = int(request_tick)
        d = int(delay_ticks)
        if not 1 <= d <= 5:
            raise ValueError("Use a declared in-distribution action delay in1..5")
        before = dict(self.counts)
        torch.cuda.synchronize()
        started = time.monotonic()
        if visual is not None:
            # Fixed-delay protocol: compute this exact frame, never a latest-ready substitute.
            self.vision(
                visual["observation"],
                visual["capture_s"],
                visual["source_tick"],
                visual["requested_tick"],
                return_features=False,
            )
        if not self.cache_meta:
            raise RuntimeError("VLM cache is empty")
        if self.cache_meta["capture_s"] > state_capture_s + 1e-6:
            raise ValueError("Future image cannot condition an older state")
        shifted = 0
        recovered = False
        if self.variant == "pir2":
            if self.anchor is None:
                raise RuntimeError("Bootstrap first")
            shifted = extra_buffer_shift(self.anchor, r)
            head = self.core.model.action_head
            with torch.inference_mode():
                if shifted >= head.config.action_horizon:
                    # A complete missed horizon has no usable rolling trajectory.
                    # Re-bootstrap from current state and actual cached VLM features;
                    # explicitly account for the extra four evaluations.
                    action_inputs, _ = self.core._collate_state_only(observation)
                    features = BatchFeature(data={**self.core._vl_cache})
                    age = max(
                        0,
                        round((state_capture_s - self.cache_meta["capture_s"]) / 0.05),
                    )
                    clean = self.core.model.forward_dit_action_only(
                        features,
                        action_inputs,
                        {
                            "force_nonstreaming": True,
                            "num_inference_timesteps": 4,
                            "image_delay": min(age, head.config.image_delay_max),
                        },
                    )
                    head.seed_streaming_buffer(clean["action_pred"], slide_steps=d)
                    recovered = True
                elif shifted:
                    buffer = head._stream_buf
                    noise = torch.randn(
                        buffer.shape[0],
                        shifted,
                        buffer.shape[2],
                        device=buffer.device,
                        dtype=buffer.dtype,
                    )
                    head._stream_buf = torch.cat([buffer[:, shifted:], noise], dim=1)
                    head._stream_buf_t = torch.cat(
                        [
                            head._stream_buf_t[:, shifted:],
                            torch.zeros(
                                buffer.shape[0],
                                shifted,
                                device=buffer.device,
                                dtype=head._stream_buf_t.dtype,
                            ),
                        ],
                        dim=1,
                    )
                prefix = np.asarray(committed_actions, dtype=np.float32)
                if prefix.shape != (d, 7):
                    raise ValueError("Incorrect committed prefix shape")
                normalized = self.core._normalize_pad_inpaint(
                    prefix, {"state": observation["state"]}
                )
                head._stream_buf[:, :d] = torch.as_tensor(
                    normalized,
                    device=head._stream_buf.device,
                    dtype=head._stream_buf.dtype,
                )[None]
                head._stream_buf_t[:, :d] = head.config.noise_s
            options = {"slide_steps": d, "num_inference_steps_per_call": 1}
        else:
            options = {"num_inference_timesteps": 4}
        actions, info = self.core.get_action_chunk_cached(
            {"state": observation["state"]},
            options=options,
            t_state_capture=state_capture_s,
            period_ms=50.0,
        )
        torch.cuda.synchronize()
        raw = self.raw(actions)
        if self.variant == "pir2":
            if not np.allclose(raw[:d], committed_actions, atol=0.01, rtol=0.01):
                raise RuntimeError("Committed prefix changed")
            self.anchor = r + d
        end = (
            2 * d
            if self.variant == "pir2" or output_scope == "next_interval"
            else len(raw)
        )
        audit = {
            "request_tick": r,
            "d_request": d,
            "valid_start_tick": r + d,
            "valid_end_tick": r + end,
            "buffer_origin_after": self.anchor,
            "extra_buffer_shift": shifted,
            "buffer_restarted": recovered,
            "state_capture_s": state_capture_s,
            "cache": dict(self.cache_meta),
            "image_delay_ticks": info["image_delay_ticks"],
            "image_delay_ms": info["image_delay_ms"],
            "server_seconds": time.monotonic() - started,
            "forward_counts": {k: self.counts[k] - before[k] for k in before},
        }
        expected = (5 if recovered else 1) if self.variant == "pir2" else 4
        if audit["forward_counts"]["dit"] != expected:
            raise RuntimeError(f"Unexpected DiT count {audit}")
        with (self.output / "requests.jsonl").open("a") as f:
            f.write(json.dumps(audit) + "\n")
        return {"actions": raw[d:end], "audit": audit}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5570)
    parser.add_argument("--variant", choices=["flow", "pir2"], required=True)
    parser.add_argument("--role", choices=["action", "vlm"], default="action")
    add_optimization_arguments(parser)
    args = parser.parse_args()
    optimization = optimization_config(args)
    optimization.validate_variant(args.variant)
    args.output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    original = policy_module.AutoModel.from_pretrained

    def checked(*pos, **kwargs):
        model, info = original(*pos, **kwargs, output_loading_info=True)
        (args.output / "weight-loading.json").write_text(
            json.dumps(info, indent=2) + "\n"
        )
        if any(
            info.get(k)
            for k in [
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            ]
        ):
            raise RuntimeError(str(info))
        return model

    policy_module.AutoModel.from_pretrained = checked
    core = DecoupledGr00tPolicy("LIBERO_PANDA", str(args.checkpoint), device="cuda:0")
    if args.role == "action":
        assert core.model.config.streaming == (args.variant == "pir2")
        assert (
            core.model.config.action_horizon
            == len(core.modality_configs["action"].delta_indices)
            == 40
        )
    backend = Backend(core, args.variant, args.output)
    server = PolicyServer(core, host="127.0.0.1", port=args.port)
    identity = {
        **checkpoint_identity(args.checkpoint),
        "variant": args.variant,
        "role": args.role,
    }
    (args.output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    server.register_endpoint("identity", lambda: identity, requires_input=False)
    server.register_endpoint("reset", backend.reset)
    server.register_endpoint("bootstrap", backend.bootstrap)
    server.register_endpoint("vision", backend.vision)
    server.register_endpoint("install", backend.install)
    server.register_endpoint("plan", backend.plan)
    with GrootOptimizations(core, optimization) as optimized:
        if optimization.enabled:
            identity["inference_optimization"] = optimized.evidence()
            (args.output / "identity.json").write_text(
                json.dumps(identity, indent=2) + "\n"
            )
        print(
            "TIMED_PROTOCOL_SERVER_READY",
            args.variant,
            args.role,
            args.port,
            flush=True,
        )
        server.run()


if __name__ == "__main__":
    main()
