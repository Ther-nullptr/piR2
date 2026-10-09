"""Reusable GR00T LIBERO backend with private CUDA streams for each worker."""

import hashlib
import json
import re
import time

import numpy as np
from libero_protocol_scheduler import extra_buffer_shift, project_commands

KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def endpoint_layout(device, vlm_device, port, vlm_port, role):
    """Validate explicit logical CUDA devices; physical mapping comes from the mask."""
    devices = [device] + ([vlm_device] if vlm_device is not None else [])
    if any(not re.fullmatch(r"cuda:[0-9]+", value) for value in devices):
        raise ValueError("Explicit CUDA devices are required")
    if (vlm_device is None) != (vlm_port is None):
        raise ValueError("Paired VLM device and port must be supplied together")
    if vlm_device is not None and (role != "action" or vlm_port == port):
        raise ValueError("Paired mode needs an action endpoint and distinct VLM port")
    ports = [port] + ([vlm_port] if vlm_port is not None else [])
    if any(not 1 <= value <= 65535 for value in ports):
        raise ValueError("Invalid TCP port")
    layout = [{"device": device, "port": port, "role": role}]
    if vlm_device is not None:
        layout.append({"device": vlm_device, "port": vlm_port, "role": "vlm"})
    return layout


class Backend:
    def __init__(self, core, variant, output):
        import torch
        from transformers import BatchFeature

        self.torch = torch
        self.BatchFeature = BatchFeature
        self.stream = torch.cuda.Stream(device=core.model.device)
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
        self.torch.manual_seed(int((options or {}).get("seed", 1000)))
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
        self.stream.synchronize()
        started = time.monotonic()
        result = self.core.update_vlm_cache(
            observation, return_features=return_features, t_image_capture=capture_s
        )
        self.stream.synchronize()
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
            t = self.torch.from_numpy(v).to(self.core.model.device)
            if t.dtype.is_floating_point:
                t = t.to(self.core.model.dtype)
            features[k] = t
        self.stream.synchronize()
        cid, _ = self.core._prime_vlm_cache(
            self.BatchFeature(features), meta["capture_s"]
        )
        self.cache_meta = dict(meta)
        return {"cache_id": cid}

    def bootstrap(self, observation, capture_s, delay_ticks, seed=1000):
        self.reset({"seed": seed})
        before = dict(self.counts)
        self.stream.synchronize()
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
        self.stream.synchronize()
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
        self.stream.synchronize()
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
            with self.torch.inference_mode():
                if shifted >= head.config.action_horizon:
                    # A complete missed horizon has no usable rolling trajectory.
                    # Re-bootstrap from current state and actual cached VLM features;
                    # explicitly account for the extra four evaluations.
                    action_inputs, _ = self.core._collate_state_only(observation)
                    features = self.BatchFeature(data={**self.core._vl_cache})
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
                    noise = self.torch.randn(
                        buffer.shape[0],
                        shifted,
                        buffer.shape[2],
                        device=buffer.device,
                        dtype=buffer.dtype,
                    )
                    head._stream_buf = self.torch.cat(
                        [buffer[:, shifted:], noise], dim=1
                    )
                    head._stream_buf_t = self.torch.cat(
                        [
                            head._stream_buf_t[:, shifted:],
                            self.torch.zeros(
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
                head._stream_buf[:, :d] = self.torch.as_tensor(
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
        self.stream.synchronize()
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
