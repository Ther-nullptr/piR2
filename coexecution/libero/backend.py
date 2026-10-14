"""Reusable GR00T LIBERO backend with private CUDA streams for each worker."""

import hashlib
import json
import re
import time
from copy import deepcopy

import numpy as np

from coexecution.libero.features import (
    TIMING_VERSION,
    decode_features,
    encode_features,
    feature_evidence,
    validate_feature_meta,
    validate_feature_options,
    validate_features,
)
from scripts.libero_protocol_scheduler import extra_buffer_shift, project_commands

KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def action_execution_counts(before, after, graph_before, graph_after):
    """Keep Python observations separate from verified Flow4 graph evaluations."""
    physical = {key: after[key] - before[key] for key in before}
    logical = dict(physical)
    graph, source = None, "python_forward_hooks"
    if graph_before is not None or graph_after is not None:
        if graph_before is None or graph_after is None:
            raise RuntimeError("Action graph scope changed during a request")
        graph = {
            key + "_delta": graph_after[key + "_count"] - graph_before[key + "_count"]
            for key in ("capture", "replay", "failure", "logical_nfe")
        }
        if (
            graph["failure_delta"] != 0
            or graph["capture_delta"] < 0
            or graph["replay_delta"] < 1
            or graph["logical_nfe_delta"] != 4 * graph["replay_delta"]
        ):
            raise RuntimeError(f"Invalid action graph execution evidence: {graph}")
        logical["dit"] = graph["logical_nfe_delta"]
        source = "captured_flow4_graph"
    return {
        "forward_counts": logical,
        "python_forward_counts": physical,
        "dit_count_source": source,
        "action_graph": graph,
    }


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
    def __init__(
        self,
        core,
        variant,
        output,
        feature_transport="numpy_fp32",
        *,
        feature_exchange=None,
        role="action",
        paired=False,
    ):
        validate_feature_options(feature_transport)
        if feature_transport == "cuda_local" and (
            feature_exchange is None or not paired
        ):
            raise ValueError("cuda_local requires a paired feature exchange")
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
        self.cache_install_timing = {}
        self.feature_transport = feature_transport
        self.feature_exchange, self.role, self.paired = feature_exchange, role, paired
        self.counts = {"vlm": 0, "dit": 0}
        for name, module in [
            ("vlm", core.model.backbone),
            ("dit", core.model.action_head.model),
        ]:

            def hook(_m, _i, _o, key=name):
                self.counts[key] += 1

            hook.__pir2_execution_observer__ = True
            module.register_forward_hook(hook)

    def _action_graph_evidence(self):
        session = getattr(self, "optimization", None)
        graph = getattr(session, "action_graph", None)
        return None if graph is None else graph.evidence()

    def graph_counters(self):
        session = getattr(self, "optimization", None)
        dit, vlm = (
            getattr(session, "graphs", None),
            getattr(session, "static_vlm", None),
        )
        counts = {} if vlm is None else vlm.counts
        return {
            "dit": {
                key: getattr(dit, key + "_count", 0)
                for key in ("capture", "replay", "failure")
            },
            "vlm": {
                key: counts.get("graph_" + key + "s", 0)
                for key in ("capture", "replay", "failure")
            },
        }

    def graph_execution(self, before):
        after = self.graph_counters()
        evidence = {
            key + "_delta": {
                role: after[role][key] - before[role][key] for role in before
            }
            for key in ("capture", "replay", "failure")
        }
        return {**evidence, "cold_capture": any(evidence["capture_delta"].values())}

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
        self.cache_install_timing = {}
        if getattr(self, "role", "action") == "vlm" and getattr(self, "paired", False):
            # The vision policy is deterministic in eval mode. Its episode reset
            # must not change the independent action policy's global CUDA RNG.
            exchange = getattr(self, "feature_exchange", None)
            if exchange is not None:
                self.stream.synchronize()
                exchange.reset()
        else:
            self.torch.manual_seed(int((options or {}).get("seed", 1000)))
        return {"reset": True}

    def feature_control(self, operation, session=None):
        if self.feature_transport != "cuda_local" or self.role != "vlm":
            raise ValueError(
                "Resident feature session control belongs to the VLM endpoint"
            )
        if operation == "open":
            return self.feature_exchange.open_session(session)
        if operation == "close":
            self.stream.synchronize()
            self.feature_exchange.close_session(session)
            return self.feature_exchange.evidence()
        raise ValueError("Unknown resident feature session operation")

    def vision(
        self,
        observation,
        capture_s,
        source_tick,
        requested_tick=None,
        return_features=True,
        feature_session=None,
        released_features=None,
    ):
        validate_feature_meta({"capture_s": capture_s, "source_tick": source_tick})
        resident = self.feature_transport == "cuda_local" and return_features
        if resident:
            if self.role != "vlm":
                raise ValueError(
                    "Only the paired VLM endpoint publishes resident features"
                )
            self.feature_exchange.release(
                feature_session, [] if released_features is None else released_features
            )
        elif feature_session is not None or released_features is not None:
            raise ValueError("Feature lease controls require cuda_local publication")
        before = dict(self.counts)
        execution_before = self.graph_counters()
        self.stream.synchronize()
        started = time.monotonic()
        result = self.core.update_vlm_cache(
            observation,
            return_features=return_features and self.feature_transport == "numpy_fp32",
            t_image_capture=capture_s,
        )
        self.stream.synchronize()
        if return_features and self.feature_transport == "numpy_bf16":
            with self.core._vl_cache_lock:
                if self.core._vl_cache_id != result["cache_id"]:
                    raise RuntimeError("VLM cache changed before its feature snapshot")
                result["vl_embeds"] = encode_features(
                    self.core._vl_cache, self.feature_transport, torch_module=self.torch
                )
        meta = {
            "capture_s": capture_s,
            "source_tick": source_tick,
            "requested_tick": source_tick if requested_tick is None else requested_tick,
            "frame_hashes": self.frame_hashes(observation),
            "vlm_seconds": time.monotonic() - started,
            "vlm_forward_calls": self.counts["vlm"] - before["vlm"],
            "graph_execution": self.graph_execution(execution_before),
        }
        if resident:
            with self.core._vl_cache_lock:
                if self.core._vl_cache_id != result["cache_id"]:
                    raise RuntimeError("VLM cache changed before resident publication")
                result["vl_embeds"], meta = self.feature_exchange.publish_cuda(
                    feature_session,
                    self.core._vl_cache,
                    meta,
                    torch_module=self.torch,
                    stream=self.stream,
                    started_s=started,
                )
            result["feature_exchange"] = self.feature_exchange.evidence()
        elif return_features:
            meta["feature_wire"] = {
                **feature_evidence(result["vl_embeds"]),
                "source": "vlm_cache_raw_tensor"
                if self.feature_transport == "numpy_bf16"
                else "upstream_return_features",
            }
        self.cache_meta = meta
        self.cache_install_timing = {}
        result["meta"] = meta
        return result

    def install(self, vl_embeds, meta):
        started = time.monotonic()
        validate_feature_meta(meta)
        imported_meta = deepcopy(meta)
        imported_meta["feature_wire"] = {
            **meta.get("feature_wire", {}),
            **feature_evidence(vl_embeds),
        }
        if getattr(self, "feature_transport", "numpy_fp32") == "cuda_local":
            with self.feature_exchange.borrow_cuda(
                vl_embeds, meta, device=self.core.model.device, stream=self.stream
            ) as features:
                self.stream.synchronize()
                cid, _ = self.core._prime_vlm_cache(
                    self.BatchFeature(features), meta["capture_s"]
                )
        else:
            validate_features(vl_embeds)
            features = decode_features(
                vl_embeds,
                torch_module=self.torch,
                device=self.core.model.device,
                legacy_dtype=self.core.model.dtype,
            )
            self.stream.synchronize()
            cid, _ = self.core._prime_vlm_cache(
                self.BatchFeature(features), meta["capture_s"]
            )
        self.cache_meta = imported_meta
        self.cache_install_timing = {
            "server_install_started_s": started,
            "server_install_completed_s": time.monotonic(),
        }
        return {"cache_id": cid, **self.cache_install_timing}

    def bootstrap(self, observation, capture_s, delay_ticks, seed=1000):
        self.reset({"seed": seed})
        before = dict(self.counts)
        graph_before = self._action_graph_evidence()
        execution_before = self.graph_counters()
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
            "graph_execution": self.graph_execution(execution_before),
            **action_execution_counts(
                before, self.counts, graph_before, self._action_graph_evidence()
            ),
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
        vl_embeds=None,
        feature_meta=None,
    ):
        request_started = time.monotonic()
        r = int(request_tick)
        d = int(delay_ticks)
        if not 1 <= d <= 5:
            raise ValueError("Use a declared in-distribution action delay in1..5")
        if (vl_embeds is None) != (feature_meta is None):
            raise ValueError(
                "Combined feature payload and metadata must be supplied together"
            )
        if visual is not None and vl_embeds is not None:
            raise ValueError("Choose an image or an encoded feature, not both")
        candidate_meta = feature_meta if vl_embeds is not None else visual
        if candidate_meta is not None:
            validate_feature_meta(
                candidate_meta, state_capture_s=state_capture_s, request_tick=r
            )
        if vl_embeds is not None:
            self.install(vl_embeds, feature_meta)
        before = dict(self.counts)
        graph_before = self._action_graph_evidence()
        execution_before = self.graph_counters()
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
        validate_feature_meta(
            self.cache_meta, state_capture_s=state_capture_s, request_tick=r
        )
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
            "timing_version": TIMING_VERSION,
            "feature_use_boundary": "server_plan_started_s",
            "server_request_started_s": request_started,
            "server_plan_started_s": started,
            "server_request_seconds": time.monotonic() - request_started,
            "server_install_started_s": self.cache_install_timing.get(
                "server_install_started_s"
            ),
            "server_install_completed_s": self.cache_install_timing.get(
                "server_install_completed_s"
            ),
            "feature_installed_in_request": vl_embeds is not None,
            "graph_execution": self.graph_execution(execution_before),
            **action_execution_counts(
                before, self.counts, graph_before, self._action_graph_evidence()
            ),
        }
        expected = (5 if recovered else 1) if self.variant == "pir2" else 4
        if audit["forward_counts"]["dit"] != expected:
            raise RuntimeError(f"Unexpected DiT count {audit}")
        with (self.output / "requests.jsonl").open("a") as f:
            f.write(json.dumps(audit) + "\n")
        return {"actions": raw[d:end], "audit": audit}
