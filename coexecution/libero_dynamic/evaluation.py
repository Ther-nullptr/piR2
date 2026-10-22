"""Default GR00T evaluation on the pinned physical moving-object benchmark.

The upstream runtime owns carriers, contacts, success and simulation-time video.
This module owns source selection, the RPC policy, calibrated latency and evidence.
Importing it never constructs a simulator, imports CUDA, or downloads a model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .catalog import LEVELS, SPEEDS_MPS, SUITES, build_inventory
from .world import NativeFixtureTiming, observation_registry, observe_objects

TIMING_RULES = ("release_continuous_sync", "paper_quantized_sync")
OBSERVATION_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _distinct(values, allowed, name):
    if not values or len(set(values)) != len(values) or not set(values) <= set(allowed):
        raise ValueError(f"{name} must select distinct values from {tuple(allowed)}")


def _speed(value):
    if isinstance(value, bool) or value not in SPEEDS_MPS:
        raise ValueError(
            f"Dynamic speed must be one of {SPEEDS_MPS}; zero is not a static switch"
        )
    return float(value)


def select_conditions(inventory, *, suite, task_ids, levels, traj_indices, speed_mps):
    """Resolve an exact, explicit condition set; missing members are errors."""
    _distinct([suite], SUITES, "suite")
    _distinct(task_ids, range(10), "task_ids")
    _distinct(levels, LEVELS, "levels")
    _distinct(traj_indices, range(5), "traj_indices")
    speed_mps = _speed(speed_mps)
    selected = [
        copy.deepcopy(row)
        for row in inventory["conditions"]
        if row["suite"] == suite
        and row["task_id"] in task_ids
        and row["level"] in levels
        and row["traj_index"] in traj_indices
        and row["speed_mps"] == speed_mps
    ]
    expected = len(task_ids) * len(levels) * len(traj_indices)
    if len(selected) != expected or len({r["item_id"] for r in selected}) != expected:
        raise ValueError("Reviewed inventory does not provide exact selected coverage")
    return sorted(
        selected, key=lambda row: (row["level"], row["task_id"], row["traj_index"])
    )


def override_motion_speed(config, speed_mps):
    """Copy and replace every alpha_mps, preserving angular motion and geometry."""
    speed_mps = _speed(speed_mps)
    result, changes = copy.deepcopy(config), []

    def visit(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "alpha_mps":
                    changes.append(
                        {"path": [*path, key], "before": value, "after": speed_mps}
                    )
                    node[key] = speed_mps
                else:
                    visit(value, [*path, key])
        elif isinstance(node, list):
            for index, value in enumerate(node):
                visit(value, [*path, index])

    visit(result, [])
    if not changes:
        raise ValueError("Reviewed configuration contains no alpha_mps fields")
    return result, changes


def applied_latency(raw_seconds, timing_rule, control_freq=20):
    if (
        isinstance(raw_seconds, bool)
        or not isinstance(raw_seconds, (int, float))
        or not math.isfinite(raw_seconds)
        or raw_seconds < 0
    ):
        raise ValueError("Latency must be finite nonnegative seconds")
    if timing_rule not in TIMING_RULES or control_freq != 20:
        raise ValueError("Use a declared dynamic timing rule at 20 Hz")
    if timing_rule == "paper_quantized_sync":
        return math.ceil(raw_seconds * control_freq) / control_freq
    return float(raw_seconds)


def _pick(mapping, keys):
    if mapping is None:
        return None
    return {key: copy.deepcopy(mapping[key]) for key in keys if key in mapping}


def _stable_source(source):
    result = _pick(
        source, ("revision", "source_sha256", "manifest_sha256", "module_sha256")
    )
    if result is not None and "submodules" in source:
        result["submodules"] = sorted(
            (
                _pick(item, ("name", "revision", "header_sha256"))
                for item in source["submodules"]
            ),
            key=lambda item: item.get("name", ""),
        )
    return result


def _stable_backend(backend):
    if backend is None:
        return None
    result = _pick(
        backend,
        (
            "backend_id",
            "backend_revision",
            "device_arch",
            "quant_format",
            "operator_profile",
            "required_operators",
        ),
    )
    result["runtime"] = _pick(
        backend.get("runtime"), ("torch", "cuda", "triton", "device_name")
    )
    result["source"] = _stable_source(backend.get("source"))
    binary = backend.get("binary") or {}
    candidates = list(binary.get("loaded_candidates") or [])
    if binary.get("selected"):
        candidates.append(binary["selected"])
    # Attribution as newly/pre-loaded and cache filenames are process history.
    # Retain every observed content identity when selection is ambiguous.
    result["binary_sha256"] = sorted(
        {row["sha256"] for row in candidates if row.get("sha256")}
    )
    return result


def _stable_optimization(optimization):
    result = _pick(
        optimization,
        ("config", "coverage", "operator_selection", "dit_graph_superseded"),
    )
    result["backend"] = _stable_backend(optimization.get("backend"))
    adapters = {
        "image_array_preparation": ("active", "representation"),
        "input_preparation": ("backbone_keys", "action_keys"),
        "tokenizer_cache": ("processors", "max_entries"),
        "dit_graph": ("max_entries",),
        "action_graph": ("active", "max_entries", "logical_nfe_per_replay"),
        "static_vlm": (
            "active",
            "max_entries",
            "pixel_outputs_cached",
            "unsupported_policy",
            "metadata_validity",
            "vision_rope_binding",
        ),
        "compiled_host_guards": ("backend", "source_sha256", "binary_sha256"),
    }
    for name, keys in adapters.items():
        result[name] = _pick(optimization.get(name), keys)
    action = optimization.get("action_graph")
    if action is not None:
        result["action_graph"]["kv_reuse"] = _pick(
            action.get("kv_reuse"),
            ("enabled", "cross_attention_layers", "projections_per_body"),
        )
    static = optimization.get("static_vlm")
    if static is not None:
        result["static_vlm"]["vlm_graph"] = _pick(
            static.get("vlm_graph"), ("enabled", "max_entries")
        )
    return result


def _stable_runtime_receipt(receipt):
    if receipt is None:
        return None
    # identity_sha256 includes receipt inventories; instead select stable facts.
    # Full receipts (including graph/cache observations) remain in worker_identity.
    result = _pick(
        receipt, ("model_contract", "configuration_sha256", "operator_profile_sha256")
    )
    result["source_manifest_sha256"] = (receipt.get("source") or {}).get(
        "manifest_sha256"
    )
    result["model_source_manifest_sha256"] = (receipt.get("model_source") or {}).get(
        "manifest_sha256"
    )
    result["device"] = _pick(
        receipt.get("device"),
        ("name", "compute_capability", "uuid_sha256", "torch", "cuda"),
    )
    result["backend"] = _stable_backend(receipt.get("backend"))
    return result


def worker_binding(identity, expected_checkpoint, precision):
    """Bind calibration to stable execution facts, excluding process history.

    Expected path is checked for admission; hashes identify checkpoint content
    across relocation. Optional receipt fields remain explicitly unobserved when
    absent. Enable the model server's --runtime-receipt for source/runtime binding.
    """
    checkpoint = identity.get("checkpoint")
    if (
        not checkpoint
        or Path(checkpoint).resolve() != Path(expected_checkpoint).resolve()
    ):
        raise ValueError("Worker checkpoint differs from --expected-checkpoint")
    optimization = identity.get("inference_optimization") or {}
    actual_precision = optimization.get("config", {}).get("precision", "bf16")
    if actual_precision != precision:
        raise ValueError(
            f"Worker precision is {actual_precision}, expected {precision}"
        )
    hashes = identity.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Worker must report checkpoint SHA256 evidence")
    if (
        identity.get("variant") != "flow"
        or identity.get("role") != "action"
        or identity.get("predicted_horizon") != 40
        or identity.get("decoded_horizon") != 16
    ):
        raise ValueError(
            "Default evaluator requires ordinary Flow4 predicted H40 and native decoded H16"
        )
    keys = (
        "sha256",
        "variant",
        "role",
        "predicted_horizon",
        "decoded_horizon",
        "physical_device_uuid",
        "paired_same_process",
        "server_layout",
        "feature_transport",
        "capture_coordination",
        "stream_local_synchronization",
        "vision_reset_rng",
        "timing_version",
    )
    binding = {key: copy.deepcopy(identity.get(key)) for key in keys}
    binding["schema"] = "pir2.libero_dynamic.execution_signature.v1"
    binding["precision"] = actual_precision
    binding["inference_optimization"] = _stable_optimization(optimization)
    binding["runtime_receipt"] = _stable_runtime_receipt(
        identity.get("runtime_receipt")
    )
    return _jsonable(binding)


def observation_fingerprint(observation, language):
    """Hash the actual native policy input, including camera bytes and instruction."""
    digest = hashlib.sha256(language.encode("utf-8"))
    for key in OBSERVATION_KEYS:
        if key not in observation:
            continue
        array = np.ascontiguousarray(observation[key])
        digest.update(key.encode("utf-8"))
        digest.update(str((array.dtype.str, array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def has_cold_capture(record):
    audit = record.get("server_audit", {})
    return bool(
        (audit.get("action_graph") or {}).get("capture_delta", 0)
        or (audit.get("graph_execution") or {}).get("cold_capture", False)
        or (audit.get("vlm_graph_execution") or {}).get("cold_capture", False)
    )


def make_calibration(
    samples, *, warmup_requests, binding, scope, max_p95_ratio, worker_identity=None
):
    if (
        type(warmup_requests) is not int
        or warmup_requests < 1
        or isinstance(max_p95_ratio, bool)
        or not math.isfinite(max_p95_ratio)
        or max_p95_ratio < 1
    ):
        raise ValueError("Calibration needs warmup and a finite stability bound >=1")
    samples = [applied_latency(value, TIMING_RULES[0]) for value in samples]
    if not samples or min(samples) <= 0:
        raise ValueError("Calibration requires positive retained full-request samples")
    median = float(np.median(samples))
    p95 = float(np.percentile(samples, 95))
    stable = p95 / median <= max_p95_ratio
    return {
        "schema": "pir2.libero_dynamic.calibration.v1",
        "kind": "measured",
        "status": "stable" if stable else "unstable",
        "boundary": "full get_action_chunk: preprocessing + RPC + postprocessing; warmup/reset excluded",
        "statistic": "median",
        "raw_latency_s": median,
        "mean_s": float(np.mean(samples)),
        "p95_s": p95,
        "max_p95_over_median": float(max_p95_ratio),
        "observed_p95_over_median": p95 / median,
        "warmup_requests": int(warmup_requests),
        "retained_requests": len(samples),
        "calibration_protocol": "benchmark_8_warmup_88_retained"
        if warmup_requests >= 8 and len(samples) >= 88
        else "diagnostic",
        "samples_s": samples,
        "worker_binding": copy.deepcopy(binding),
        "worker_identity": copy.deepcopy(worker_identity),
        "scope": copy.deepcopy(scope),
    }


def validate_calibration(record, *, binding, suite, request_contract=None):
    if (
        record.get("schema") != "pir2.libero_dynamic.calibration.v1"
        or record.get("kind") != "measured"
        or record.get("status") != "stable"
    ):
        raise ValueError("A stable measured calibration file is required")
    if record.get("worker_binding") != binding:
        raise ValueError("Calibration worker identity differs from the actual worker")
    scope = record.get("scope", {})
    if scope.get("suite") != suite:
        raise ValueError("Calibration suite differs from evaluation suite")
    if not scope.get("observation_sha256") or not scope.get("capture_scope"):
        raise ValueError("Calibration must identify the observed inputs and scope")
    if (
        request_contract is not None
        and scope.get("request_contract") != request_contract
    ):
        raise ValueError(
            "Calibration request contract differs from actual camera/state shapes or action prefix"
        )
    rebuilt = make_calibration(
        record.get("samples_s", []),
        warmup_requests=record.get("warmup_requests", 0),
        binding=binding,
        scope=scope,
        max_p95_ratio=record.get("max_p95_over_median", 2),
    )
    if (
        rebuilt["status"] != "stable"
        or rebuilt["raw_latency_s"] != record.get("raw_latency_s")
        or rebuilt["retained_requests"] != record.get("retained_requests")
        or rebuilt["calibration_protocol"] != record.get("calibration_protocol")
        or record.get("warmup_requests", 0) < 1
    ):
        raise ValueError(
            "Calibration samples/counts do not prove a stable frozen latency"
        )
    return rebuilt["raw_latency_s"]


def counterfactual_latency(seconds):
    return {
        "schema": "pir2.libero_dynamic.calibration.v1",
        "kind": "counterfactual",
        "status": "declared",
        "raw_latency_s": applied_latency(seconds, TIMING_RULES[0]),
        "retained_requests": 0,
        "warmup_requests": 0,
        "calibration_protocol": "counterfactual",
        "samples_s": [],
        "scope": {"capture_scope": "explicit scalar; not a measured model speed"},
    }


def calibration_summary(record):
    """Carry the actual calibration population into every run summary."""
    names = {
        "calibration_kind": "kind",
        "calibration_protocol": "calibration_protocol",
        "warmup_requests": "warmup_requests",
        "retained_requests": "retained_requests",
    }
    return {
        name: None if record is None else record[key] for name, key in names.items()
    }


@contextmanager
def owned_policy_factory(legacy, policy):
    """Temporarily supply this episode's policy, with unconditional restoration."""
    original = legacy.build_policy

    def build(args, env, item):
        if args.policy != "groot_rpc":
            raise ValueError("Owned dynamic evaluator only accepts groot_rpc")
        return policy

    legacy.build_policy = build
    try:
        yield
    finally:
        legacy.build_policy = original


def close_controller(controller, *, timeout_s):
    """Drain and close the socket on its executor, then shut down that executor."""
    errors = []
    deadline = time.monotonic() + timeout_s
    future = controller.future
    if future is not None:
        try:
            future.result(timeout=max(0, deadline - time.monotonic()))
        except Exception as error:  # noqa: BLE001 -- preserve any worker error while still closing its socket
            errors.append(f"draining inference: {type(error).__name__}: {error}")
    try:
        closed = controller.executor.submit(controller.policy.close)
        closed.result(timeout=max(0, deadline - time.monotonic()))
    except Exception as error:  # noqa: BLE001 -- close failures are persisted by the caller
        errors.append(f"closing policy: {type(error).__name__}: {error}")
    finally:
        # A timed-out close stays queued behind its running RPC, retaining socket
        # affinity. Never destroy the socket from this control thread.
        controller.executor.shutdown(wait=False, cancel_futures=False)
    return errors


def attach_capture(snapshot, packet, *, capture_s):
    return {
        **snapshot,
        "__pir2_capture": {
            "source_tick": int(packet.step_idx),
            "source_tick_kind": "action_step",
            "sim_time_s": float(packet.sim_time_s),
            "capture_s": float(capture_s),
        },
    }


def add_evaluate_arguments(parser):
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--task-ids", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS))
    parser.add_argument("--traj-indices", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--speed-mps", type=float, choices=SPEEDS_MPS, required=True)
    parser.add_argument("--model-host", default="127.0.0.1")
    parser.add_argument("--model-port", type=int, required=True)
    parser.add_argument("--expected-checkpoint", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "w8a8"), required=True)
    parser.add_argument("--chunk-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--timing-rule", choices=TIMING_RULES, default=TIMING_RULES[0])
    latency = parser.add_mutually_exclusive_group(required=True)
    latency.add_argument("--calibrate", action="store_true")
    latency.add_argument("--calibration", type=Path)
    latency.add_argument("--latency-seconds", type=float)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--calibration-requests", type=int, default=88)
    parser.add_argument("--calibration-max-p95-ratio", type=float, default=2)
    parser.add_argument("--rpc-timeout-seconds", type=int, default=900)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--video-fps", type=float, default=60)
    parser.add_argument(
        "--save-videos", action=argparse.BooleanOptionalAction, default=True
    )


def _validate_options(args):
    for name in ("max_steps", "warmup_requests", "calibration_requests", "camera_size"):
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not 1 <= args.chunk_steps <= 16:
        raise ValueError("Default GR00T uses native decoded H16; choose prefix 1..16")
    if args.seed != 0 or args.init_state_id != 0:
        raise ValueError(
            "Reviewed benchmark currently requires seed=0 and init_state_id=0"
        )
    if not 1 <= args.rpc_timeout_seconds <= (2**31 - 1) // 1000:
        raise ValueError("RPC timeout must fit positive int32 milliseconds")
    if not 1 <= args.model_port <= 65535:
        raise ValueError("Invalid model port")
    if not math.isfinite(args.video_fps) or not 0 < args.video_fps <= 500:
        raise ValueError("Video FPS must be finite and in (0,500]")
    if (
        not math.isfinite(args.calibration_max_p95_ratio)
        or args.calibration_max_p95_ratio < 1
    ):
        raise ValueError("Calibration p95/median bound must be finite and >=1")
    if not args.expected_checkpoint.is_dir():
        raise ValueError("--expected-checkpoint must be an existing local checkpoint")
    if args.latency_seconds is not None:
        applied_latency(args.latency_seconds, args.timing_rule)


def _load_runtime(upstream):
    """Import only the selected pinned environment in this dedicated process."""
    upstream = Path(upstream).resolve()
    for name in ("libero", "realtime_eval", "eval_dynamic_libero_style"):
        module = sys.modules.get(name)
        if module is not None and not Path(module.__file__).resolve().is_relative_to(
            upstream
        ):
            raise ValueError(
                f"{name} was already loaded from another checkout; start a fresh process"
            )
    sys.path[:0] = [str(upstream / "scripts"), str(upstream)]
    runner = importlib.import_module("realtime_eval.runner")
    runtime = importlib.import_module("realtime_eval.runtime")
    for module in (runner, runtime, runtime.legacy):
        if not Path(module.__file__).resolve().is_relative_to(upstream):
            raise ValueError("Dynamic runtime import escaped the pinned checkout")
    return SimpleNamespace(runner=runner, runtime=runtime, legacy=runtime.legacy)


def _source_arguments(source, options):
    args = source.runner.build_parser().parse_args(
        ["--output-dir", str(options.output)]
    )
    args.policy = "groot_rpc"
    args.control_mode = "sync_hold"
    args.prefetch_threshold = 0
    args.replan_steps = options.chunk_steps
    args.max_steps = options.max_steps
    args.seed = options.seed
    args.init_state_id = options.init_state_id
    args.levels = options.levels
    args.traj_indices = options.traj_indices
    args.stop_on_success = True
    args.latency_source = "fixed"
    args.calibrated_latency_seconds = 0.0  # Replaced before the first scored request.
    args.save_videos = options.save_videos
    args.save_review_video = False
    args.save_model_input_videos = True
    args.model_input_video_layout = "dual"
    args.model_input_video_size = options.camera_size
    args.save_wrist_input_videos = True
    args.save_step_trace = True
    args.camera_size = options.camera_size
    args.video_fps = options.video_fps
    args.reviewed_old_physics = True
    source.legacy.apply_reviewed_old_physics_profile(args)
    return args


def _effective_items(source, args, options, conditions):
    import yaml

    config_dir = options.upstream / "libero/libero/dynamic_configs"
    paths = source.legacy.write_effective_config_files(
        config_dir, options.output, args.levels, args
    )
    changes = {}
    for level, name in paths.items():
        path = Path(name)
        config, edits = override_motion_speed(
            yaml.safe_load(path.read_text()), options.speed_mps
        )
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        changes[level] = {
            "path": name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "edits": edits,
        }
    _write_json(
        options.output / "speed-override.json",
        {"speed_mps": options.speed_mps, "configs": changes},
    )
    # build_items must READ the overridden copies, not the original config_dir.
    items = source.legacy.build_items(
        Path(next(iter(paths.values()))).parent,
        args.levels,
        args.traj_indices,
        keep_static_support_contact=args.keep_static_support_contact,
        static_support_clearance_m=args.static_support_clearance_m,
        config_path_by_level=paths,
    )
    by_id = {item["item_id"]: item for item in items}
    if any(row["item_id"] not in by_id for row in conditions):
        raise ValueError("Runtime condition coverage differs from reviewed inventory")
    return [by_id[row["item_id"]] for row in conditions]


def make_episode_type(source, options, session):
    """Extend setup/evidence hooks while retaining upstream physical control."""
    runtime = source.runtime

    class GrootEpisode(runtime.RealtimeEpisode):
        def __init__(self, item, args, *, policy):
            # One process, one episode: source seed + worker_slot remains seed 0.
            super().__init__(item, args, worker_slot=0)
            self.policy = policy
            self.controller = None
            self.env = None
            self.initial_observation = None
            self.last_capture_s = None
            self.control_started = False
            self.rollout_sim_time_s = 0.0
            self.motion_events = []
            self.chunk_events = []
            self.cleanup_errors = []
            self.calibration_scope = None
            self.applied_latency_s = None
            self.closed = False
            self.action_step = 0
            self.active_chunk_id = None
            self.video_samples = []
            self.next_video_sample_s = 1.0 / options.video_fps
            self.fixture_timing = None
            self.latency_backend = (
                "source_preview_exact_hold" if self.use_preview else None
            )
            self.observed_objects = []
            self._source_action_callback_offset_s = 0.0

        def _motion_phase_s(self):
            if self.fixture_timing is not None:
                return float(self.env.dynamic_fixture_body_state["elapsed"])
            return float(self.preview_elapsed)

        def _motion_event(self, kind, **fields):
            if self.env is None:
                return
            poses = observe_objects(
                self.env, self.observed_objects, runtime._object_pose
            )
            self.motion_events.append(
                {
                    "kind": kind,
                    "wall_s": time.monotonic(),
                    "rollout_sim_time_s": self.rollout_sim_time_s,
                    "motion_phase_s": self._motion_phase_s(),
                    "physics_sim_time_s": float(self.env.sim.data.time),
                    "target_object_poses": poses,
                    "action_step": self.action_step,
                    "active_chunk_id": self.active_chunk_id,
                    **fields,
                }
            )

        def _setup_world(self, env, init_state):
            self.env = env
            result = super()._setup_world(env, init_state)
            if result[3] != 40:
                raise ValueError("Require 40 actual source pre-policy control steps")
            if not self.use_preview:
                # Preserve source-native fixture initialization and env.step.
                # Its pending-latency path is extracted below, never forced into
                # the carrier-only preview setup.
                self.fixture_timing = NativeFixtureTiming(env)
                self.latency_backend = self.fixture_timing.backend
            self.observed_objects = observation_registry(
                self.item, env, runtime._target_entries(self.item, env)
            )
            self.initial_observation = result[0]
            # _step_env timestamps the actual returned observation. The fallback
            # is only for an alternative source setup returning its own capture.
            if self.last_capture_s is None:
                self.last_capture_s = time.monotonic()
            self._motion_event("initial_observation", capture_s=self.last_capture_s)
            return result

        def _step_env(self, env, action, **kwargs):
            started = time.monotonic()
            physics_started = float(env.sim.data.time)
            if self.control_started:
                callback = kwargs.get("on_substep")
                offset = self._source_action_callback_offset_s
                self._source_action_callback_offset_s = 0.0
                if callback is not None and offset:
                    # Native run() expects hold physics inside this action and
                    # subtracts tau from its video base. Hold has already run,
                    # so correct only its callback, not our physical frame map.
                    kwargs = {**kwargs, "on_substep": lambda dt: callback(dt + offset)}
                kwargs = self._recording_kwargs(kwargs, "action")
            result = super()._step_env(env, action, **kwargs)
            self.last_capture_s = time.monotonic()
            if self.control_started:
                advanced = float(env.sim.data.time) - physics_started
                if not math.isclose(advanced, self.preview_dt, rel_tol=0, abs_tol=1e-8):
                    raise RuntimeError(
                        "Action physics duration differs from one control step"
                    )
                self.rollout_sim_time_s += advanced
                self._motion_event(
                    "action_completed",
                    physics_elapsed_s=advanced,
                    action=np.asarray(action).copy(),
                    wall_started_s=started,
                    capture_s=self.last_capture_s,
                )
                self.action_step += 1
            return result

        def _apply_sync_latency(self, env, duration, **kwargs):
            self._motion_event("hold_started", applied_latency_s=float(duration))
            recording = self._recording_kwargs(kwargs, "inference_hold")
            if self.fixture_timing is not None:
                advanced = self.fixture_timing.advance(duration, **recording)
                self._source_action_callback_offset_s += float(advanced)
            else:
                advanced = super()._apply_sync_latency(env, duration, **recording)
            self.rollout_sim_time_s += float(advanced)
            self._motion_event("hold_completed", applied_latency_s=float(advanced))
            return advanced

        def _recording_kwargs(self, kwargs, phase):
            callback = kwargs.get("on_substep")
            if callback is None or not options.save_videos:
                return kwargs
            base_sim_time_s = self.rollout_sim_time_s

            def record(advanced_s):
                absolute_sim_time_s = base_sim_time_s + float(advanced_s)
                callback(advanced_s)
                # Source callback samples at this threshold after a real physics
                # substep. Verify the sample count against its final manifest.
                if absolute_sim_time_s + 1e-9 >= self.next_video_sample_s:
                    self.video_samples.append(
                        {
                            "frame_index": len(self.video_samples),
                            "media_time_s": len(self.video_samples) / options.video_fps,
                            "nominal_sim_time_s": self.next_video_sample_s,
                            "sample_sim_time_s": absolute_sim_time_s,
                            "wall_completed_s": time.monotonic(),
                            "phase": phase,
                            "action_step": self.action_step,
                            "active_chunk_id": self.active_chunk_id,
                        }
                    )
                    self.next_video_sample_s += 1.0 / options.video_fps

            return {**kwargs, "on_substep": record}

        def _call(self, controller, function):
            return controller.executor.submit(function).result(
                timeout=options.rpc_timeout_seconds + 5
            )

        def _sample_policy(self, controller, observation, language, phase):
            begin = len(self.policy.records)

            def infer():
                started = time.perf_counter()
                self.policy.get_action_chunk(observation, language)
                return time.perf_counter() - started

            try:
                elapsed = self._call(controller, infer)
            finally:
                for record in self.policy.records[begin:]:
                    record["phase"] = phase
                    record["item_id"] = self.item["item_id"]
            if phase == "calibration" and any(
                has_cold_capture(record) for record in self.policy.records[begin:]
            ):
                raise ValueError(
                    "Cold CUDA graph capture in retained calibration; increase warmup"
                )
            return elapsed

        def _close_owned(self):
            if not self.closed and self.controller is not None:
                self.closed = True
                self.cleanup_errors.extend(
                    close_controller(
                        self.controller, timeout_s=options.rpc_timeout_seconds + 5
                    )
                )

        def _build_controller(self, policy, language):
            controller = super()._build_controller(policy, language)
            self.controller = controller
            controller.close = self._close_owned
            try:
                identity = self._call(controller, policy.prepare)
                binding = worker_binding(
                    identity, options.expected_checkpoint, options.precision
                )
                if session["binding"] is not None and session["binding"] != binding:
                    raise ValueError(
                        "Worker identity changed between selected episodes"
                    )
                session["binding"] = binding
                packet = SimpleNamespace(step_idx=0, sim_time_s=0.0)
                observation = attach_capture(
                    self.initial_observation, packet, capture_s=self.last_capture_s
                )
                self.calibration_scope = {
                    "suite": options.suite,
                    "item_id": self.item["item_id"],
                    "capture_scope": "episode_initial_observation",
                    "observation_sha256": observation_fingerprint(
                        observation, language
                    ),
                    "observation_count": 1,
                    "language": language,
                    "source_tick": 0,
                    "source_tick_kind": "action_step",
                    "rollout_sim_time_s": 0.0,
                    "motion_phase_s": self._motion_phase_s(),
                    "pre_policy_dynamic_steps": 40,
                    "request_contract": {
                        "chunk_steps": policy.chunk_steps,
                        "observations": {
                            key: {
                                "shape": list(np.asarray(observation[key]).shape),
                                "dtype": np.asarray(observation[key]).dtype.str,
                            }
                            for key in OBSERVATION_KEYS
                            if key in observation
                        },
                    },
                }
                for _ in range(options.warmup_requests):
                    self._sample_policy(controller, observation, language, "warmup")
                if session["calibration"] is None:
                    samples = []
                    for _ in range(options.calibration_requests):
                        samples.append(
                            self._sample_policy(
                                controller, observation, language, "calibration"
                            )
                        )
                    session["calibration"] = make_calibration(
                        samples,
                        warmup_requests=options.warmup_requests,
                        binding=binding,
                        worker_identity=identity,
                        scope={
                            **self.calibration_scope,
                            "capture_scope": "first_selected_initial_observation",
                        },
                        max_p95_ratio=options.calibration_max_p95_ratio,
                    )
                    _write_json(
                        options.output / "calibration.json", session["calibration"]
                    )
                calibration = session["calibration"]
                if calibration["kind"] == "measured":
                    raw_latency = validate_calibration(
                        calibration,
                        binding=binding,
                        suite=options.suite,
                        request_contract=self.calibration_scope["request_contract"],
                    )
                else:
                    raw_latency = calibration["raw_latency_s"]
                self.applied_latency_s = applied_latency(
                    raw_latency, options.timing_rule
                )
                controller.latency_sampler.fixed_seconds = self.applied_latency_s
                self.args.calibrated_latency_seconds = self.applied_latency_s
                snapshot = controller.snapshot_request_observation

                def owned_snapshot(packet, actions):
                    self._motion_event(
                        "observation_submitted",
                        source_tick=packet.step_idx,
                        source_tick_kind="action_step",
                        capture_s=self.last_capture_s,
                        observation_sim_time_s=packet.sim_time_s,
                    )
                    return attach_capture(
                        snapshot(packet, actions), packet, capture_s=self.last_capture_s
                    )

                controller.snapshot_request_observation = owned_snapshot
                reset = controller.reset

                def owned_reset():
                    reset()
                    # Reset the formal episode RNG once after all warmup calls,
                    # on the same connection; exclude reset/identity RPCs from tau.
                    self._call(controller, policy.prepare)
                    self.control_started = True

                controller.reset = owned_reset
                tick = controller.tick

                def owned_tick(packet, actions):
                    begin = len(policy.records)
                    try:
                        ready = tick(packet, actions)
                    finally:
                        for record in policy.records[begin:]:
                            record["phase"] = "scored"
                            record["item_id"] = self.item["item_id"]
                            record["cold_capture"] = has_cold_capture(record)
                    for chunk in ready:
                        self.active_chunk_id = chunk.chunk_id
                        chunk.metadata.update(
                            {
                                "timing_rule": options.timing_rule,
                                "raw_calibration_latency_s": raw_latency,
                                "applied_latency_s": self.applied_latency_s,
                                "sim_latency_s": self.applied_latency_s,
                                "physics_time_quantized_to_control_steps": options.timing_rule
                                == "paper_quantized_sync",
                                "latency_application": options.timing_rule,
                                "policy_request_sequence": policy.records[-1][
                                    "request_sequence"
                                ],
                            }
                        )
                        self.chunk_events.append(
                            {
                                "chunk_id": chunk.chunk_id,
                                "obs_step_idx": chunk.obs_step_idx,
                                "wall_s": time.monotonic(),
                                "rollout_sim_time_s": packet.sim_time_s,
                                "policy_elapsed_s": chunk.policy_elapsed_s,
                                "metadata": copy.deepcopy(chunk.metadata),
                            }
                        )
                    return ready

                controller.tick = owned_tick
                return controller
            except BaseException:
                self._close_owned()
                raise

        def run(self):
            try:
                with owned_policy_factory(source.legacy, self.policy):
                    result = super().run()
            finally:
                self._close_owned()
                base = options.output / self.item["level"] / self.item["suite"]
                evidence = base / "policy_audits" / f"{self.item['item_id']}.json"
                _write_json(
                    evidence,
                    {
                        "schema": "pir2.libero_dynamic.policy_audit.v1",
                        "item_id": self.item["item_id"],
                        "identity": self.policy.identity,
                        "calibration": session["calibration"],
                        "initial_observation": self.calibration_scope,
                        "records": self.policy.records,
                        "chunks": self.chunk_events,
                        "motion_events": self.motion_events,
                        "observation_registry": self.observed_objects,
                        "latency_backend": self.latency_backend,
                        "fixture_timing": None
                        if self.fixture_timing is None
                        else self.fixture_timing.records,
                        "fixture_driver": None
                        if self.fixture_timing is None
                        else self.fixture_timing.driver_evidence,
                        "video_samples": self.video_samples,
                        "cleanup_errors": self.cleanup_errors,
                        "clocks": {
                            "wall_s": "monotonic process clock",
                            "rollout_sim_time_s": "physics time since first policy observation; includes inference holds",
                            "source_tick": "action count; excludes hold substeps",
                        },
                    },
                )
            result.update(
                {
                    "evaluation_schema": "pir2.libero_dynamic.default.v1",
                    "policy": "groot_rpc",
                    "protocol": "default_vla_flow4",
                    "precision": options.precision,
                    "timing_rule": options.timing_rule,
                    "speed_mps": options.speed_mps,
                    "policy_identity": self.policy.identity,
                    "raw_calibration_latency_s": None
                    if session["calibration"] is None
                    else session["calibration"]["raw_latency_s"],
                    "applied_latency_s": self.applied_latency_s,
                    "calibration_kind": None
                    if session["calibration"] is None
                    else session["calibration"]["kind"],
                    "calibration_scope": None
                    if session["calibration"] is None
                    else session["calibration"].get("scope"),
                    "calibration_protocol": None
                    if session["calibration"] is None
                    else session["calibration"]["calibration_protocol"],
                    "calibration_warmup_requests": None
                    if session["calibration"] is None
                    else session["calibration"]["warmup_requests"],
                    "calibration_retained_requests": None
                    if session["calibration"] is None
                    else session["calibration"]["retained_requests"],
                    "warmup_requests_per_episode": options.warmup_requests,
                    "video_view_contract": "Upstream agentview/wrist observer renders during real physics substeps; not postprocessed GR00T request tensors",
                    "video_time_mapping": str(evidence) + "#video_samples",
                    "policy_audit": str(evidence),
                    "cleanup_errors": self.cleanup_errors,
                    "scored_cold_captures": sum(
                        record.get("cold_capture", False)
                        for record in self.policy.records
                        if record.get("phase") == "scored"
                    ),
                    "latency_backend": self.latency_backend,
                    "fixture_driver": None
                    if self.fixture_timing is None
                    else self.fixture_timing.driver_evidence,
                    "world_contract": (
                        "Physical carriers move throughout controls and robot-held inference latency. "
                        if self.use_preview
                        else "Source-native fixture body controls; owned exact-duration robot-held latency, "
                        "with final 10 ms fixture update block shortened to the actual remainder. "
                    )
                    + (
                        "Latency uses exact calibrated seconds."
                        if options.timing_rule == TIMING_RULES[0]
                        else "Latency is ceil(raw_tau * 20) / 20 seconds, applied once."
                    ),
                }
            )
            if self.cleanup_errors:
                result["status"] = "error"
                result["error"] = "; ".join(self.cleanup_errors)
            if options.save_videos and result.get("video_frame_count") != len(
                self.video_samples
            ):
                result["status"] = "error"
                result["error"] = (
                    "Video sample mapping differs from recorded frame count"
                )
            # Upstream hides media paths on errors; retain any partial recordings
            # explicitly, without describing an incomplete episode as success.
            result["recorded_media"] = [
                str(path)
                for folder in ("model_input_videos", "wrist_input_videos", "videos")
                for path in (base / folder).glob(f"{self.item['item_id']}--*.mp4")
                if path.is_file()
            ]
            _write_json(Path(result["manifest"]), result)
            return result

    return GrootEpisode


def evaluate_default(options):
    """Evaluate a declared moving condition set with one existing Flow4 worker."""
    from .policy import GrootChunkPolicy

    _validate_options(options)
    options.upstream = options.upstream.expanduser().resolve()
    options.output = options.output.expanduser().resolve()
    options.expected_checkpoint = options.expected_checkpoint.expanduser().resolve()
    inventory = build_inventory(options.upstream, suites=[options.suite])
    if (
        inventory["upstream"]["actual_revision"]
        != inventory["upstream"]["expected_revision"]
    ):
        raise ValueError("Evaluation requires the actual pinned upstream Git revision")
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(options.upstream),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    if dirty.stdout.strip():
        raise ValueError("Pinned upstream contains tracked modifications")
    conditions = select_conditions(
        inventory,
        suite=options.suite,
        task_ids=options.task_ids,
        levels=options.levels,
        traj_indices=options.traj_indices,
        speed_mps=options.speed_mps,
    )
    if options.calibration is not None:
        calibration = json.loads(options.calibration.read_text())
        if (
            not isinstance(calibration, dict)
            or calibration.get("schema") != "pir2.libero_dynamic.calibration.v1"
            or calibration.get("kind") != "measured"
            or calibration.get("status") != "stable"
        ):
            raise ValueError(
                "--calibration requires a stable measured calibration, not a scalar counterfactual"
            )
    elif options.latency_seconds is not None:
        calibration = counterfactual_latency(options.latency_seconds)
    else:
        calibration = None
    options.output.mkdir(parents=True, exist_ok=False)
    _write_json(
        options.output / "evaluation.json",
        {
            "schema": "pir2.libero_dynamic.default.v1",
            "status": "starting",
            "configuration": vars(options),
            "upstream": inventory["upstream"],
            "conditions": conditions,
            "protocol": "default_vla_flow4",
            "phase_contract": inventory["phase_contract"],
            "calibration_policy": "one frozen latency for selected batch; calibrate first selected initial observation",
            "action_budget_contract": "max_steps counts actions only; inference holds add physical time",
        },
    )
    if calibration is not None:
        _write_json(options.output / "calibration.json", calibration)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    session = {"calibration": calibration, "binding": None}
    results = []
    try:
        source = _load_runtime(options.upstream)
        args = _source_arguments(source, options)
        items = _effective_items(source, args, options, conditions)
        _write_json(options.output / "items.json", items)
        _write_json(options.output / "runtime-config.json", vars(args))
        episode_type = make_episode_type(source, options, session)
        for index, item in enumerate(items):
            print(
                f"[{index + 1}/{len(items)}] {item['item_id']} {options.precision}",
                flush=True,
            )
            policy = GrootChunkPolicy(
                host=options.model_host,
                port=options.model_port,
                chunk_steps=options.chunk_steps,
                seed=options.seed,
                timeout_ms=options.rpc_timeout_seconds * 1000,
            )
            result = episode_type(item, args, policy=policy).run()
            result["condition_id"] = conditions[index]["condition_id"]
            result["upstream"] = inventory["upstream"]
            _write_json(Path(result["manifest"]), result)
            results.append(result)
            source.runner.write_summary(options.output, results)
            summary_path = options.output / "summary.json"
            _write_json(
                summary_path,
                {
                    **json.loads(summary_path.read_text()),
                    **calibration_summary(session["calibration"]),
                },
            )
            print(
                f"status={result['status']} success={result['success']} steps={result['steps']}",
                flush=True,
            )
            if result["status"] != "ok":
                raise RuntimeError(
                    f"Dynamic episode failed: {result.get('error')}; retained {result['manifest']}"
                )
        summary = {
            "schema": "pir2.libero_dynamic.evaluation_summary.v1",
            "status": "complete",
            "protocol": "default_vla_flow4",
            "timing_rule": options.timing_rule,
            "precision": options.precision,
            "suite": options.suite,
            "conditions": len(results),
            "successes": sum(row["success"] for row in results),
            "success_rate": sum(row["success"] for row in results) / len(results),
            **calibration_summary(session["calibration"]),
            "raw_calibration_latency_s": session["calibration"]["raw_latency_s"],
            "applied_latency_s": applied_latency(
                session["calibration"]["raw_latency_s"], options.timing_rule
            ),
            "output": str(options.output),
        }
        _write_json(options.output / "evaluation-summary.json", summary)
        return summary
    except BaseException as error:
        _write_json(
            options.output / "failure.json",
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "completed_episodes": len(results),
                "calibration": session["calibration"],
            },
        )
        raise
