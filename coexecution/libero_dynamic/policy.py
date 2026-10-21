"""Default fresh-VLM/Flow4 GR00T bridge for the dynamic benchmark controller.

The controller owns physics and latency admission. This adapter owns only its
thread-affine RPC connection, observation conversion, and the native action prefix.
"""

import math
import threading
import time
from copy import deepcopy
from itertools import pairwise
from types import SimpleNamespace

import numpy as np

from coexecution.libero.synchronous import validate_synchronous_identity


def prepare_groot_observation(raw_observation, language):
    """Reuse official camera rotation/state conventions without creating an env."""
    from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv as OfficialLiberoEnv

    from scripts.libero_wallclock import to_nested

    # The official quaternion conversion clips its input in-place. Preserve the
    # benchmark snapshot for provenance and other consumers of the observation.
    observation = dict(raw_observation)
    observation["robot0_eef_quat"] = np.array(
        raw_observation["robot0_eef_quat"], copy=True
    )
    holder = SimpleNamespace(_task_description=language)
    return to_nested(OfficialLiberoEnv._process_observation(holder, observation))


def _client_factory(**options):
    from gr00t.policy.server_client import PolicyClient

    class ExplicitCloseClient(PolicyClient):
        def close(self):
            self.socket.close(linger=0)
            self.context.term()

        def __del__(self):
            # Upstream's destructor touches the socket. GC may run on the main
            # thread after executor shutdown; cleanup belongs to explicit close.
            pass

    return ExplicitCloseClient(**options)


class GrootChunkPolicy:
    """An ordinary full-policy request, followed by K native decoded commands.

    Construction and reset perform no RPC. The first get_action_chunk creates
    the client on the inference executor; all later calls and close must run on
    that same thread. Drain requests and submit close to that executor BEFORE
    shutting down the upstream controller, whose close does not close its policy.

    Optional raw observation ``__pir2_capture`` metadata contains source_tick,
    sim_time_s and/or capture_s (monotonic seconds). Without a physical source
    tick, the RPC source_tick is a request sequence, explicitly labelled in the
    records; it must not be interpreted as a simulator tick.
    """

    supports_rtc_conditioning = False
    required_observation_history_frames = 1
    required_action_history_steps = 0

    def __init__(
        self,
        host="127.0.0.1",
        port=5570,
        *,
        chunk_steps=8,
        seed=1000,
        timeout_ms=120000,
        client_factory=None,
        preprocessor=None,
    ):
        if type(chunk_steps) is not int or not 1 <= chunk_steps <= 40:
            raise ValueError("Action prefix must be an integer in [1,40]")
        if type(seed) is not int or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Invalid policy port")
        if type(timeout_ms) is not int or not 0 < timeout_ms < 2**31:
            raise ValueError("RPC timeout must be positive int32 milliseconds")
        self.chunk_steps = chunk_steps
        self.seed = seed
        self.records = []
        self.identity = None
        self._options = {"host": host, "port": port, "timeout_ms": timeout_ms}
        self._factory = client_factory or _client_factory
        self._preprocess = preprocessor or prepare_groot_observation
        self._client = None
        self._owner = None
        self._lock = threading.Lock()
        self._closed = False
        self._generation = 0
        self._applied_generation = None
        self._request_sequence = 0

    def reset(self, seed=None):
        """Queue one deterministic server reset for the next inference call."""
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("Seed must be a nonnegative integer")
        with self._lock:
            if self._closed:
                raise RuntimeError("Policy is closed")
            if seed is not None:
                self.seed = seed
            self._generation += 1

    def _connection(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("Policy is closed")
            owner = threading.current_thread()
            if self._owner is not None and self._owner is not owner:
                raise RuntimeError("RPC operations require the owning thread")
            self._owner = owner
        if self._client is None:
            self._client = self._factory(**self._options)
        if self.identity is None:
            identity = self._client.call_endpoint("identity", requires_input=False)
            validate_synchronous_identity(identity, self.chunk_steps)
            if identity.get("predicted_horizon") != 40:
                raise ValueError("Default GR00T policy requires predicted H40")
            self.identity = deepcopy(identity)
        return self._client

    @staticmethod
    def _capture(raw_observation, request_sequence, started):
        metadata = raw_observation.get("__pir2_capture", {})
        if not isinstance(metadata, dict):
            raise TypeError("Observation capture metadata must be a mapping")
        source_tick = metadata.get("source_tick", request_sequence)
        capture_s = metadata.get("capture_s", started)
        sim_time_s = metadata.get("sim_time_s")
        if type(source_tick) is not int or source_tick < 0:
            raise ValueError("Observation source tick must be a nonnegative integer")
        if not isinstance(capture_s, (int, float)) or not math.isfinite(capture_s):
            raise ValueError("Observation capture time must be finite")
        if sim_time_s is not None and (
            not isinstance(sim_time_s, (int, float))
            or not math.isfinite(sim_time_s)
            or sim_time_s < 0
        ):
            raise ValueError("Simulation time must be finite and nonnegative")
        return {
            "source_tick": source_tick,
            "source_tick_kind": (
                "physical_tick" if "source_tick" in metadata else "request_sequence"
            ),
            "capture_s": capture_s,
            "capture_time_kind": (
                "observation_capture" if "capture_s" in metadata else "policy_entry"
            ),
            "sim_time_s": sim_time_s,
        }

    def _validate_result(self, model_actions, audit, capture):
        if (
            model_actions.ndim != 2
            or model_actions.shape[1] != 7
            or not np.isfinite(model_actions).all()
            or len(model_actions) != self.identity["decoded_horizon"]
            or audit.get("decoded_horizon") != len(model_actions)
            or len(model_actions) < self.chunk_steps
        ):
            raise ValueError("Expected finite native decoded (T,7) model actions")
        if not np.isin(model_actions[:, 6], [0, 1]).all():
            raise ValueError("Expected GR00T gripper in projected 0/1 convention")
        if audit.get("forward_counts") != {"vlm": 1, "dit": 4}:
            raise ValueError("Default plan requires one fresh VLM and four DiT calls")
        if any(audit.get(key) != capture[key] for key in ("capture_s", "source_tick")):
            raise ValueError("Plan does not identify the requested fresh observation")
        stages = [
            audit.get(key)
            for key in (
                "vlm_started_s",
                "vlm_completed_s",
                "action_started_s",
                "action_completed_s",
            )
        ]
        if any(not isinstance(t, (float, int)) or not math.isfinite(t) for t in stages):
            raise ValueError("Invalid server stage timestamps")
        if any(a > b for a, b in pairwise(stages)):
            raise ValueError("VLM and Flow4 server stages must be ordered")

    def get_action_chunk(self, raw_observation, language):
        """Return raw LIBERO commands; record full decoded output before slicing."""
        started = time.monotonic()
        if not isinstance(language, str) or not language.strip():
            raise ValueError("Task language must be a nonempty string")
        client = self._connection()
        with self._lock:
            generation, seed = self._generation, self.seed
            request_sequence = self._request_sequence
            self._request_sequence += 1
        capture = self._capture(raw_observation, request_sequence, started)
        record = {
            "request_sequence": request_sequence,
            "generation": generation,
            "seed": seed,
            "request_started_s": started,
            **capture,
            "execution_prefix_steps": self.chunk_steps,
            "predicted_horizon": self.identity["predicted_horizon"],
            "decoded_horizon": self.identity["decoded_horizon"],
            "status": "pending",
        }
        try:
            if self._applied_generation != generation:
                record["reset_started_s"] = time.monotonic()
                client.call_endpoint("reset", {"options": {"seed": seed}})
                record["reset_completed_s"] = time.monotonic()
                self._applied_generation = generation
            observation = self._preprocess(raw_observation, language)
            record["rpc_started_s"] = time.monotonic()
            try:
                response = client.call_endpoint(
                    "synchronous_plan",
                    {
                        "observation": observation,
                        "capture_s": capture["capture_s"],
                        "source_tick": capture["source_tick"],
                    },
                )
            finally:
                record["rpc_completed_s"] = time.monotonic()
            audit = deepcopy(response["audit"])
            record["server_audit"] = audit
            model_actions = np.array(response["actions"], dtype=np.float32, copy=True)
            record["model_actions"] = model_actions
            self._validate_result(model_actions, audit, capture)
            actions = model_actions[: self.chunk_steps].copy()
            actions[:, 6] = 1 - 2 * actions[:, 6]
            with self._lock:
                if generation != self._generation:
                    raise RuntimeError(
                        "Policy reset during inference; discard old episode result"
                    )
                record["actions"] = actions.copy()
                record["status"] = "complete"
            return actions
        except Exception as error:
            record["status"] = "error"
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["request_completed_s"] = time.monotonic()
            with self._lock:
                self.records.append(record)

    def close(self):
        """Close on the inference executor before that executor shuts down."""
        with self._lock:
            if self._closed:
                return
            if (
                self._owner is not None
                and self._owner is not threading.current_thread()
            ):
                raise RuntimeError("RPC close requires the owning thread")
            self._closed = True
        if self._client is not None:
            self._client.close()
            self._client = None
