"""Thread-owned real πR² RPCs; no physical-world or logical-clock advancement."""

import hashlib
import math
import threading
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from coexecution.libero.features import validate_features
from coexecution.libero_dynamic.policy import (
    _client_factory,
    prepare_groot_observation,
)


def _positive_capture(value):
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError("Logical captures require a finite positive epoch")


def _actions(value, length):
    result = np.array(value, dtype=np.float32, copy=True)
    if result.shape != (length, 7) or not np.isfinite(result).all():
        raise ValueError("Expected finite native decoded actions of declared length")
    if not np.isin(result[:, 6], (0, 1)).all():
        raise ValueError("Expected projected GR00T 0/1 gripper commands")
    return result


class Pir2RPC:
    """Two independent policies, called serially by one control-thread owner.

    ``bootstrap``/``vision`` receive raw observations and reuse the official
    preprocessing. ``plan`` installs only the explicitly supplied published
    feature, in the same RPC as inference. The S2 policy never shares S1 cache
    state. Actual monotonic request times and server audit are returned untouched;
    callers must not subtract logical image captures from those wall times.
    """

    def __init__(
        self,
        *,
        action_port,
        vision_port,
        expected_checkpoint,
        precision,
        host="127.0.0.1",
        timeout_ms=120000,
        client_factory=None,
        preprocessor=None,
    ):
        if any(
            type(p) is not int or not 1 <= p <= 65535
            for p in (action_port, vision_port)
        ):
            raise ValueError("RPC ports must be integers in 1..65535")
        if action_port == vision_port:
            raise ValueError("S1 and S2 require independent endpoints")
        if type(timeout_ms) is not int or not 0 < timeout_ms < 2**31:
            raise ValueError("Invalid RPC timeout")
        if precision not in ("bf16", "w8a8"):
            raise ValueError("Dynamic πR² supports declared BF16/W8A8 only")
        self.expected_checkpoint = Path(expected_checkpoint).resolve()
        if not self.expected_checkpoint.is_dir():
            raise ValueError("Expected checkpoint must already exist locally")
        self.precision = precision
        self._options = {
            role: {"host": host, "port": port, "timeout_ms": timeout_ms}
            for role, port in (("action", action_port), ("vlm", vision_port))
        }
        self._factory = client_factory or _client_factory
        self._preprocess = preprocessor or prepare_groot_observation
        self._clients = {}
        self._owner = None
        self._closed = False
        self.identities = None
        self.records = []
        self._anchor = None

    def _thread(self):
        if self._closed:
            raise RuntimeError("πR² RPC adapter is closed")
        current = threading.current_thread()
        if self._owner is not None and self._owner is not current:
            raise RuntimeError("RPC operations require the owning thread")
        self._owner = current

    def prepare(self):
        """Validate identities without executing a model or changing model RNG."""
        self._ensure_prepared()
        return deepcopy(self.identities)

    def _ensure_prepared(self):
        """Validate once; retain ownership checks without exporting metadata."""
        self._thread()
        if self.identities is not None:
            return
        identities = {}
        for role, options in self._options.items():
            if role not in self._clients:
                self._clients[role] = self._factory(**options)
            identity = self._clients[role].call_endpoint(
                "identity", requires_input=False
            )
            actual_precision = (
                (identity.get("inference_optimization") or {})
                .get("config", {})
                .get("precision", "bf16")
            )
            if (
                identity.get("variant") != "pir2"
                or identity.get("role") != role
                or identity.get("predicted_horizon") != 40
                or identity.get("decoded_horizon") != 40
                or identity.get("feature_transport") != "numpy_bf16"
                or "combined" not in identity.get("feature_rpc_modes", [])
                or not {"bootstrap", "plan"}.issubset(
                    identity.get("execution_endpoints", [])
                )
                or actual_precision != self.precision
                or not identity.get("checkpoint")
                or Path(identity["checkpoint"]).resolve() != self.expected_checkpoint
                or not identity.get("policy_instance")
                or not isinstance(identity.get("sha256"), dict)
                or not identity["sha256"]
            ):
                raise ValueError(
                    "Incompatible πR² worker identity/checkpoint/precision"
                )
            identities[role] = deepcopy(identity)
        if (
            identities["action"]["policy_instance"]
            == identities["vlm"]["policy_instance"]
        ):
            raise ValueError("S1 and S2 must use independent policy instances")
        if identities["action"]["sha256"] != identities["vlm"]["sha256"]:
            raise ValueError("S1 and S2 checkpoint content differs")
        if self._clients["action"] is self._clients["vlm"]:
            raise ValueError("S1 and S2 require independent client connections")
        self.identities = identities

    def _call(self, role, endpoint, payload):
        self._ensure_prepared()
        record = {
            "endpoint": endpoint,
            "role": role,
            "wall_clock": "host_monotonic",
            "wall_started_s": time.monotonic(),
        }
        try:
            result = deepcopy(self._clients[role].call_endpoint(endpoint, payload))
            record["status"] = "complete"
            return result, record
        except Exception as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["wall_completed_s"] = time.monotonic()
            record["wall_duration_s"] = (
                record["wall_completed_s"] - record["wall_started_s"]
            )
            self.records.append(record)

    def _observation(self, raw, language):
        if not isinstance(language, str) or not language.strip():
            raise ValueError("Task language must be nonempty")
        return self._preprocess(deepcopy(raw), language)

    def bootstrap(self, raw, language, *, capture_s, delay_ticks, seed):
        self._thread()
        self._anchor = None
        _positive_capture(capture_s)
        if type(delay_ticks) is not int or delay_ticks not in range(1, 6):
            raise ValueError("Bootstrap delay must be 1..5")
        if type(seed) is not int or seed < 0:
            raise ValueError("Bootstrap seed must be a nonnegative integer")
        self._ensure_prepared()
        # Reset S2 first: a split server may reseed its RNG. S1 bootstrap then
        # resets its own seed and streaming state, so its first action is stable.
        self._clients["vlm"].call_endpoint("reset", {"options": {"seed": seed}})
        observation = self._observation(raw, language)
        result, wall = self._call(
            "action",
            "bootstrap",
            {
                "observation": observation,
                "capture_s": capture_s,
                "delay_ticks": delay_ticks,
                "seed": seed,
            },
        )
        if result.get("forward_counts") != {"vlm": 1, "dit": 4}:
            raise ValueError("Bootstrap requires real VLM1/DiT4 forward evidence")
        receipt = {
            "actions": _actions(result["actions"], 40),
            "capture_s": capture_s,
            "meta": {
                "capture_s": capture_s,
                "source_tick": 0,
                "requested_tick": 0,
                "frame_hashes": {
                    name: hashlib.sha256(value.tobytes()).hexdigest()
                    for name, value in observation.get("video", {}).items()
                },
            },
            "server_audit": {k: v for k, v in result.items() if k != "actions"},
            "wall": deepcopy(wall),
            "seed": seed,
            "delay_ticks": delay_ticks,
        }
        self._anchor = 0
        return receipt

    def vision(self, raw, language, *, capture_s, source_tick):
        _positive_capture(capture_s)
        result, wall = self._call(
            "vlm",
            "vision",
            {
                "observation": self._observation(raw, language),
                "capture_s": capture_s,
                "source_tick": source_tick,
                "return_features": True,
            },
        )
        meta = result.get("meta", {})
        if meta.get("capture_s") != capture_s or meta.get("source_tick") != source_tick:
            raise ValueError("S2 result does not belong to the admitted observation")
        if meta.get("vlm_forward_calls") != 1:
            raise ValueError("S2 requires one real VLM forward")
        if result.get("vl_embeds") is None:
            raise ValueError("S2 requires an independent numpy_bf16 feature snapshot")
        encoding, _ = validate_features(result["vl_embeds"])
        if encoding != "numpy_bf16":
            raise ValueError("S2 must publish the declared numpy_bf16 wire")
        return {**result, "wall": deepcopy(wall)}

    def plan(
        self,
        raw,
        language,
        *,
        request_tick,
        delay_ticks,
        committed_actions,
        state_capture_s,
        feature,
    ):
        self._thread()
        _positive_capture(state_capture_s)
        if (
            self._anchor is None
            or type(request_tick) is not int
            or request_tick < self._anchor
        ):
            raise ValueError("S1 requires bootstrap and a valid rolling request tick")
        if type(delay_ticks) is not int or delay_ticks not in range(1, 6):
            raise ValueError("S1 delay must be an integer in 1..5")
        prefix = _actions(committed_actions, delay_ticks)
        observation = self._observation(raw, language)
        meta = feature["meta"]
        if meta["capture_s"] > state_capture_s or meta["source_tick"] > request_tick:
            raise ValueError("S1 cannot use a future visual capture")
        payload = {
            "observation": {"state": observation["state"]},
            "request_tick": request_tick,
            "delay_ticks": delay_ticks,
            "committed_actions": prefix,
            "state_capture_s": state_capture_s,
        }
        if feature.get("vl_embeds") is not None:
            payload.update(vl_embeds=feature["vl_embeds"], feature_meta=deepcopy(meta))
        # A lost response or failed audit cannot establish the server's new
        # rolling state. Only a fresh successful bootstrap can recover it.
        previous_anchor = self._anchor
        self._anchor = None
        result, wall = self._call("action", "plan", payload)
        audit = result.get("audit", {})
        expected = {
            "request_tick": request_tick,
            "d_request": delay_ticks,
            "valid_start_tick": request_tick + delay_ticks,
            "valid_end_tick": request_tick + 2 * delay_ticks,
            "buffer_origin_after": request_tick + delay_ticks,
            "state_capture_s": state_capture_s,
        }
        if any(audit.get(k) != v for k, v in expected.items()):
            raise ValueError("S1 result violates its original request/valid interval")
        if any(
            audit.get("cache", {}).get(k) != meta[k]
            for k in ("capture_s", "source_tick")
        ):
            raise ValueError("S1 used an unselected/future visual feature")
        recovered = audit.get("buffer_restarted")
        if type(recovered) is not bool:
            raise ValueError("S1 must report normal/recovery execution")
        shift = request_tick - previous_anchor
        if audit.get("extra_buffer_shift") != shift or recovered != (shift >= 40):
            raise ValueError("S1 recovery contradicts its rolling buffer history")
        if audit.get("forward_counts") != {"vlm": 0, "dit": 5 if recovered else 1}:
            raise ValueError("S1 requires real normal1/recovery5 DiT forward evidence")
        if audit.get("image_delay_ticks") != max(
            0, round((state_capture_s - meta["capture_s"]) / 0.05)
        ):
            raise ValueError("Server image age must use the supplied logical captures")
        validated = {
            "actions": _actions(result["actions"], delay_ticks),
            "audit": deepcopy(audit),
            "wall": deepcopy(wall),
        }
        self._anchor = request_tick + delay_ticks
        return validated

    def close(self):
        if self._closed:
            return
        self._thread()
        errors = []
        for client in self._clients.values():
            try:
                client.close()
            except Exception as error:  # noqa: BLE001 -- close both, then raise together
                errors.append(error)
        self._closed = True
        if errors:
            raise RuntimeError(f"RPC cleanup failed: {errors}")
