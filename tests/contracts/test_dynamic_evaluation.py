"""Dynamic evaluation contracts; no simulator or neural model is constructed."""

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from coexecution.libero_dynamic import evaluation as ev


def identity(checkpoint, precision="bf16"):
    return {
        "checkpoint": str(checkpoint),
        "sha256": {"model.safetensors": "a" * 64},
        "variant": "flow",
        "role": "action",
        "predicted_horizon": 40,
        "decoded_horizon": 16,
        "server_instance": "server-a",
        "policy_instance": "policy-a",
        "physical_device_uuid": "gpu-a",
        "inference_optimization": {"config": {"precision": precision}},
    }


def test_nested_speed_override_preserves_raw_geometry_and_angular_motion():
    original = {
        "defaults": {"motion": {"alpha_mps": 0.055, "alpha_yaw": 0.2}},
        "tasks": {
            "suite": {
                0: {
                    "activation_variants": [
                        {
                            "motion": {
                                "alpha_mps": 0.07,
                                "waypoints": [[0, 1], [2, 3]],
                            },
                            "target_objects": [
                                {"motion": {"alpha_mps": 0.08, "alpha_yaw": 0.4}}
                            ],
                        }
                    ]
                }
            }
        },
    }
    before = copy.deepcopy(original)
    changed, edits = ev.override_motion_speed(original, 0.06)
    assert original == before
    assert len(edits) == 3
    variant = changed["tasks"]["suite"][0]["activation_variants"][0]
    assert variant["motion"] == {"alpha_mps": 0.06, "waypoints": [[0, 1], [2, 3]]}
    assert variant["target_objects"][0]["motion"] == {
        "alpha_mps": 0.06,
        "alpha_yaw": 0.4,
    }
    assert changed["defaults"]["motion"]["alpha_yaw"] == 0.2


@pytest.mark.parametrize("speed", [0, -0.1, float("nan"), 0.05, True])
def test_speed_rejects_false_static_or_nonbenchmark_values(speed):
    with pytest.raises(ValueError):
        ev.override_motion_speed({"alpha_mps": 0.06}, speed)


def test_condition_selection_rejects_missing_and_duplicate_requested_conditions():
    conditions = [
        {
            "suite": "libero_spatial",
            "task_id": task,
            "level": level,
            "traj_index": traj,
            "speed_mps": speed,
            "item_id": f"{task}-{level}-{traj}",
        }
        for task in range(10)
        for level in ("l1", "l2", "l3")
        for traj in range(5)
        for speed in (0.03, 0.06, 0.09)
    ]
    options = {
        "suite": "libero_spatial",
        "task_ids": [0, 3],
        "levels": ["l1", "l3"],
        "traj_indices": [0, 4],
        "speed_mps": 0.06,
    }
    selected = ev.select_conditions({"conditions": conditions}, **options)
    assert len(selected) == 8
    assert {row["task_id"] for row in selected} == {0, 3}
    with pytest.raises(ValueError, match="distinct"):
        ev.select_conditions(
            {"conditions": conditions}, **{**options, "task_ids": [0, 0]}
        )
    with pytest.raises(ValueError, match="coverage"):
        ev.select_conditions(
            {"conditions": conditions[:-3]}, **{**options, "task_ids": [9]}
        )


def test_latency_rules_preserve_raw_tau_and_never_double_quantize():
    assert ev.applied_latency(0.060, "release_continuous_sync") == 0.060
    assert ev.applied_latency(0.060, "paper_quantized_sync") == 0.100
    assert ev.applied_latency(0.100, "paper_quantized_sync") == 0.100
    assert ev.applied_latency(0, "paper_quantized_sync") == 0
    for value in (-1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            ev.applied_latency(value, "release_continuous_sync")


def test_binding_rejects_wrong_precision_or_checkpoint_and_tracks_optimization(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    worker = identity(checkpoint)
    binding = ev.worker_binding(worker, checkpoint, "bf16")
    assert binding["sha256"] == worker["sha256"]
    with pytest.raises(ValueError, match="precision"):
        ev.worker_binding(worker, checkpoint, "w8a8")
    with pytest.raises(ValueError, match="checkpoint"):
        ev.worker_binding(worker, tmp_path / "other", "bf16")
    altered = copy.deepcopy(worker)
    altered["inference_optimization"]["config"]["fusion"] = True
    assert ev.worker_binding(altered, checkpoint, "bf16") != binding


def test_calibration_is_bound_to_actual_worker_observations_and_stability(tmp_path):
    binding = ev.worker_binding(identity(tmp_path), tmp_path, "bf16")
    scope = {
        "suite": "libero_spatial",
        "observation_sha256": "d" * 64,
        "item_id": "spatial0",
        "capture_scope": "first_selected_initial_observation",
    }
    record = ev.make_calibration(
        [0.05, 0.06, 0.07],
        warmup_requests=8,
        binding=binding,
        scope=scope,
        max_p95_ratio=2,
    )
    assert record["raw_latency_s"] == 0.06
    assert record["retained_requests"] == 3
    ev.validate_calibration(record, binding=binding, suite="libero_spatial")
    with pytest.raises(ValueError, match="identity"):
        ev.validate_calibration(
            record,
            binding={**binding, "physical_device_uuid": "new"},
            suite="libero_spatial",
        )
    with pytest.raises(ValueError, match="suite"):
        ev.validate_calibration(record, binding=binding, suite="libero_goal")
    unstable = ev.make_calibration(
        [0.01, 0.01, 1],
        warmup_requests=8,
        binding=binding,
        scope=scope,
        max_p95_ratio=2,
    )
    assert unstable["status"] == "unstable"
    with pytest.raises(ValueError, match="stable"):
        ev.validate_calibration(unstable, binding=binding, suite="libero_spatial")


def test_explicit_latency_is_counterfactual_not_measured():
    record = ev.counterfactual_latency(0.06)
    assert record["kind"] == "counterfactual"
    assert record["retained_requests"] == 0
    assert record["raw_latency_s"] == 0.06


def deployment_identity(checkpoint):
    """Shape of service.py + GrootOptimizations.evidence, with synthetic hashes."""
    worker = identity(checkpoint, "w8a8")
    worker.update(
        pid=123,
        cuda_stream=987,
        logical_device="cuda:0",
        cuda_visible_devices="3",
        output="/tmp/run-a",
        startup_time=10,
    )
    worker["inference_optimization"] = {
        "config": {
            "precision": "w8a8",
            "fusion": True,
            "dit_graph": True,
            "action_graph": True,
            "vlm_graph": True,
            "operator_profile": {"sha256": "b" * 64},
        },
        "coverage": {
            "linears": ["layer.0.q_proj"],
            "category_linears": [],
            "operator_adapters": {"group_writeback": {"installed_groups": 16}},
        },
        "backend": {
            "backend_id": "pir2_integer",
            "backend_revision": "sha256:" + "c" * 64,
            "device_arch": "sm_89",
            "quant_format": "w8a8-symmetric-row-v1",
            "runtime": {
                "torch": "2.7.1+cu126",
                "cuda": "12.6",
                "triton": "3.3.1",
                "device": "cuda:0",
                "device_name": "NVIDIA RTX 6000 Ada Generation",
            },
            "source": {
                "source_sha256": "c" * 64,
                "manifest_sha256": "d" * 64,
                "module_sha256": "e" * 64,
                "module_path": "coexecution/kernels/integer.py",
                "submodules": [
                    {
                        "name": "cutlass",
                        "revision": "f" * 40,
                        "path": "/tmp/cutlass-a",
                        "header_sha256": None,
                    }
                ],
            },
            "binary": {
                "status": "observed_backend_load",
                "selected": {
                    "filename": "integer_ext.so",
                    "sha256": "1" * 64,
                    "newly_loaded": True,
                },
                "loaded_candidates": [
                    {
                        "filename": "integer_ext.so",
                        "sha256": "1" * 64,
                        "newly_loaded": True,
                    }
                ],
            },
        },
        "image_array_preparation": {
            "active": True,
            "calls": 0,
            "images": 0,
            "representation": "owned contiguous uint8 HWC RGB",
        },
        "compiled_host_guards": {
            "source_sha256": "2" * 64,
            "binary_sha256": "3" * 64,
            "backend": "cython",
            "binary": "/tmp/cache-a/guard.so",
            "build_settings": {"compiler": {"path": "/tmp/gcc-a", "sha256": "4" * 64}},
        },
        "input_preparation": {
            "backbone_keys": ["pixel_values"],
            "action_keys": ["state"],
            "fallbacks": {},
        },
        "tokenizer_cache": {
            "processors": 2,
            "entries": 0,
            "max_entries": 16,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "bypasses": 0,
        },
        "dit_graph": {
            "capture_count": 0,
            "replay_count": 0,
            "failure_count": 0,
            "eviction_count": 0,
            "resident_signatures": 0,
            "max_entries": 64,
        },
        "action_graph": {
            "capture_count": 0,
            "replay_count": 0,
            "logical_nfe_count": 0,
            "resident_signatures": 0,
            "max_entries": 16,
            "active": True,
            "logical_nfe_per_replay": 4,
            "kv_reuse": {
                "enabled": True,
                "cross_attention_layers": [0, 1],
                "projections_per_body": 4,
                "replay_count": 0,
                "last_body_proof": {"epoch": 0},
                "retained_tensors": 0,
            },
        },
        "static_vlm": {
            "active": True,
            "calls": 0,
            "cache_hits": 0,
            "preparation_host_ms": 0,
            "execution_epoch": 0,
            "max_entries": 32,
            "vlm_graph": {
                "enabled": True,
                "capture_count": 0,
                "replay_count": 0,
                "resident_signatures": 0,
                "max_entries": 32,
            },
        },
        "operator_selection": {
            "implementations": {"integer_gemm": "split_k"},
            "parameters": {"integer_gemm": {"slices": 8}},
        },
    }
    worker["runtime_receipt"] = {
        "source": {
            "manifest_sha256": "5" * 64,
            "git_commit": "6" * 40,
            "loaded_modules": {
                "coexecution.libero.backend": "/tmp/runtime-a/backend.py"
            },
        },
        "model_source": {"manifest_sha256": "7" * 64, "status": "observed"},
        "device": {
            "logical_device": "cuda:0",
            "uuid_sha256": "8" * 64,
            "name": "NVIDIA RTX 6000 Ada Generation",
            "compute_capability": [8, 9],
            "torch": "2.7.1+cu126",
            "cuda": "12.6",
            "clock_sample": None,
        },
        "configuration_sha256": "9" * 64,
        "enabled_adapters": {"dit_graph": {"capture_count": 0}},
        "identity_sha256": "0" * 64,
    }
    return worker


def test_binding_survives_restart_relocation_and_real_adapter_counter_changes(tmp_path):
    worker = deployment_identity(tmp_path / "checkpoint-a")
    moved = copy.deepcopy(worker)
    moved.update(
        checkpoint=str(tmp_path / "checkpoint-b"),
        server_instance="new-server",
        policy_instance="new-policy",
        pid=456,
        cuda_stream=111,
        logical_device="cuda:1",
        cuda_visible_devices="0,3",
        output="/tmp/run-b",
        startup_time=99,
    )
    opt = moved["inference_optimization"]
    opt["backend"]["runtime"]["device"] = "cuda:1"
    opt["backend"]["source"]["submodules"][0]["path"] = "/tmp/cutlass-b"
    opt["backend"]["binary"]["status"] = "unknown"
    opt["backend"]["binary"]["selected"] = None
    opt["backend"]["binary"]["loaded_candidates"][0].update(
        filename="/tmp/cache-b/integer_ext.so", newly_loaded=False
    )
    opt["compiled_host_guards"]["binary"] = "/tmp/cache-b/guard.so"
    opt["compiled_host_guards"]["build_settings"]["compiler"]["path"] = "/tmp/gcc-b"
    opt["image_array_preparation"].update(calls=100, images=200)
    opt["input_preparation"]["fallbacks"] = {"video": ["some_prior_input_shape"]}
    opt["tokenizer_cache"].update(
        hits=100, misses=5, entries=5, evictions=3, bypasses=1
    )
    opt["dit_graph"].update(capture_count=5, replay_count=100, resident_signatures=5)
    opt["action_graph"].update(
        capture_count=5, replay_count=100, logical_nfe_count=400, resident_signatures=5
    )
    opt["action_graph"]["kv_reuse"].update(
        replay_count=100, retained_tensors=16, last_body_proof={"epoch": 5}
    )
    opt["static_vlm"].update(
        calls=100, cache_hits=99, preparation_host_ms=120, execution_epoch=7
    )
    opt["static_vlm"]["vlm_graph"].update(
        capture_count=5, replay_count=100, resident_signatures=5
    )
    moved["runtime_receipt"]["source"].update(
        git_commit="a" * 40, loaded_modules={"extra.module": "/tmp/runtime-b/module.py"}
    )
    moved["runtime_receipt"]["device"]["logical_device"] = "cuda:1"
    moved["runtime_receipt"]["enabled_adapters"]["dit_graph"]["capture_count"] = 5
    moved["runtime_receipt"]["identity_sha256"] = "b" * 64
    assert ev.worker_binding(worker, worker["checkpoint"], "w8a8") == ev.worker_binding(
        moved, moved["checkpoint"], "w8a8"
    )


@pytest.mark.parametrize(
    "path,value",
    [
        (("sha256", "model.safetensors"), "b" * 64),
        (("physical_device_uuid",), "gpu-b"),
        (("inference_optimization", "config", "fusion"), False),
        (("inference_optimization", "backend", "runtime", "torch"), "2.9.0+cu128"),
        (("inference_optimization", "backend", "source", "source_sha256"), "a" * 64),
        (
            (
                "inference_optimization",
                "backend",
                "binary",
                "loaded_candidates",
                0,
                "sha256",
            ),
            "b" * 64,
        ),
        (
            (
                "inference_optimization",
                "operator_selection",
                "parameters",
                "integer_gemm",
                "slices",
            ),
            4,
        ),
        (("inference_optimization", "dit_graph", "max_entries"), 32),
        (("inference_optimization", "compiled_host_guards", "binary_sha256"), "c" * 64),
        (("runtime_receipt", "source", "manifest_sha256"), "d" * 64),
        (("runtime_receipt", "model_source", "manifest_sha256"), "e" * 64),
    ],
)
def test_binding_changes_when_execution_identity_changes(tmp_path, path, value):
    worker = deployment_identity(tmp_path)
    altered = copy.deepcopy(worker)
    node = altered
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    assert ev.worker_binding(worker, tmp_path, "w8a8") != ev.worker_binding(
        altered, tmp_path, "w8a8"
    )


@pytest.mark.parametrize("horizon", [8, 40, True, None])
def test_default_evaluator_rejects_non_native_h16_workers(tmp_path, horizon):
    worker = identity(tmp_path)
    worker["decoded_horizon"] = horizon
    with pytest.raises(ValueError, match="native decoded H16"):
        ev.worker_binding(worker, tmp_path, "bf16")


@pytest.mark.parametrize(
    "warmup,retained,protocol",
    [
        (1, 1, "diagnostic"),
        (8, 87, "diagnostic"),
        (7, 88, "diagnostic"),
        (8, 88, "benchmark_8_warmup_88_retained"),
        (10, 100, "benchmark_8_warmup_88_retained"),
    ],
)
def test_calibration_labels_diagnostic_sample_counts(
    tmp_path, warmup, retained, protocol
):
    worker = identity(tmp_path)
    record = ev.make_calibration(
        [0.05] * retained,
        warmup_requests=warmup,
        binding=ev.worker_binding(worker, tmp_path, "bf16"),
        scope={},
        max_p95_ratio=2,
        worker_identity=worker,
    )
    assert record["calibration_protocol"] == protocol
    assert record["retained_requests"] == retained
    assert record["warmup_requests"] == warmup
    assert record["worker_identity"] == worker
    assert "server_instance" not in record["worker_binding"]


def test_calibration_summary_preserves_imported_counts_and_rejects_relabeling(tmp_path):
    binding = ev.worker_binding(identity(tmp_path), tmp_path, "bf16")
    record = ev.make_calibration(
        [0.05],
        warmup_requests=1,
        binding=binding,
        max_p95_ratio=2,
        scope={
            "suite": "libero_spatial",
            "capture_scope": "first_selected_initial_observation",
            "observation_sha256": "d" * 64,
        },
    )
    assert ev.calibration_summary(record) == {
        "calibration_kind": "measured",
        "calibration_protocol": "diagnostic",
        "warmup_requests": 1,
        "retained_requests": 1,
    }
    restored = json.loads(json.dumps(record))
    assert ev.calibration_summary(restored) == ev.calibration_summary(record)
    restored["calibration_protocol"] = "benchmark_8_warmup_88_retained"
    with pytest.raises(ValueError, match="samples/counts"):
        ev.validate_calibration(restored, binding=binding, suite="libero_spatial")


def test_calibration_reuse_checks_camera_shapes_and_executed_prefix(tmp_path):
    binding = ev.worker_binding(identity(tmp_path), tmp_path, "bf16")
    request = {
        "chunk_steps": 8,
        "observations": {"agentview_image": {"shape": [256, 256, 3], "dtype": "|u1"}},
    }
    record = ev.make_calibration(
        [0.05] * 88,
        warmup_requests=8,
        binding=binding,
        max_p95_ratio=2,
        scope={
            "suite": "libero_spatial",
            "capture_scope": "first_selected_initial_observation",
            "observation_sha256": "d" * 64,
            "request_contract": request,
        },
    )
    ev.validate_calibration(
        record, binding=binding, suite="libero_spatial", request_contract=request
    )
    for changed in (
        {**request, "chunk_steps": 1},
        {
            "chunk_steps": 8,
            "observations": {
                "agentview_image": {"shape": [128, 128, 3], "dtype": "|u1"}
            },
        },
    ):
        with pytest.raises(ValueError, match="request contract"):
            ev.validate_calibration(
                record,
                binding=binding,
                suite="libero_spatial",
                request_contract=changed,
            )


def test_factory_is_owned_and_restored_even_when_episode_fails():
    original = lambda *args: "original"
    legacy = SimpleNamespace(build_policy=original)
    policy = object()
    with (
        pytest.raises(RuntimeError, match="episode failed"),
        ev.owned_policy_factory(legacy, policy),
    ):
        assert (
            legacy.build_policy(SimpleNamespace(policy="groot_rpc"), None, {}) is policy
        )
        with pytest.raises(ValueError, match="groot_rpc"):
            legacy.build_policy(SimpleNamespace(policy="zero"), None, {})
        raise RuntimeError("episode failed")
    assert legacy.build_policy is original


def test_close_drains_and_closes_on_inference_owner_before_shutdown():
    calls = []
    pool = ThreadPoolExecutor(max_workers=1)

    def infer():
        calls.append(("infer", threading.get_ident()))
        return 1

    def close():
        calls.append(("close", threading.get_ident()))

    controller = SimpleNamespace(
        executor=pool, future=pool.submit(infer), policy=SimpleNamespace(close=close)
    )
    assert ev.close_controller(controller, timeout_s=2) == []
    assert [name for name, _ in calls] == ["infer", "close"]
    assert calls[0][1] == calls[1][1] != threading.get_ident()
    with pytest.raises(RuntimeError, match="shutdown"):
        pool.submit(lambda: None)


def test_close_still_closes_after_inference_failure():
    pool = ThreadPoolExecutor(max_workers=1)
    closed = []

    def fail():
        raise ValueError("inference failure")

    controller = SimpleNamespace(
        executor=pool,
        future=pool.submit(fail),
        policy=SimpleNamespace(close=lambda: closed.append(True)),
    )
    assert "inference failure" in ev.close_controller(controller, timeout_s=2)[0]
    assert closed == [True]


def test_capture_preserves_action_counter_separate_from_physical_elapsed_time():
    snapshot = {"agentview_image": np.zeros((2, 2, 3), dtype=np.uint8)}
    packet = SimpleNamespace(step_idx=8, sim_time_s=0.46)
    result = ev.attach_capture(snapshot, packet, capture_s=123.0)
    assert result["__pir2_capture"] == {
        "source_tick": 8,
        "source_tick_kind": "action_step",
        "sim_time_s": 0.46,
        "capture_s": 123.0,
    }
    assert "__pir2_capture" not in snapshot


def test_observation_fingerprint_covers_images_state_and_language():
    obs = {
        "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
    }
    base = ev.observation_fingerprint(obs, "pick cup")
    assert base == ev.observation_fingerprint(copy.deepcopy(obs), "pick cup")
    assert base != ev.observation_fingerprint(obs, "pick bowl")
    obs["robot0_eef_pos"][0] = 0.1
    assert base != ev.observation_fingerprint(obs, "pick cup")


def test_cold_capture_detection_covers_whole_action_and_component_graphs():
    assert ev.has_cold_capture(
        {"server_audit": {"graph_execution": {"cold_capture": True}}}
    )
    assert ev.has_cold_capture({"server_audit": {"action_graph": {"capture_delta": 1}}})
    assert not ev.has_cold_capture(
        {"server_audit": {"action_graph": {"capture_delta": 0}}}
    )


def test_cli_requires_explicit_latency_choice_without_importing_simulator(
    monkeypatch, capsys
):
    from coexecution.libero_dynamic.__main__ import main

    with pytest.raises(SystemExit) as caught:
        main(
            [
                "evaluate",
                "--upstream",
                "/tmp/source",
                "--output",
                "/tmp/out",
                "--suite",
                "libero_spatial",
                "--expected-checkpoint",
                "/tmp/model",
                "--precision",
                "bf16",
                "--model-port",
                "19170",
                "--speed-mps",
                "0.06",
            ]
        )
    assert caught.value.code == 2
    assert "--calibrate" in capsys.readouterr().err


def test_effective_items_read_overridden_copy_not_original_yaml(tmp_path):
    import yaml

    upstream = tmp_path / "source"
    config_dir = upstream / "libero/libero/dynamic_configs"
    config_dir.mkdir(parents=True)
    original = config_dir / "libero_dynamic_l1.yaml"
    original.write_text("motion:\n  alpha_mps: 0.055\n  alpha_yaw: 0.2\n")
    before = original.read_bytes()
    output = tmp_path / "output"

    def write(config, output_dir, levels, args):
        assert config == config_dir
        target = output_dir / "effective_dynamic_configs/libero_dynamic_l1.yaml"
        target.parent.mkdir(parents=True)
        target.write_bytes(original.read_bytes())
        return {"l1": str(target)}

    def build(config, levels, traj_indices, **kwargs):
        assert config != config_dir
        assert yaml.safe_load((config / original.name).read_text())["motion"] == {
            "alpha_mps": 0.09,
            "alpha_yaw": 0.2,
        }
        return [{"item_id": "selected"}, {"item_id": "other"}]

    source = SimpleNamespace(
        legacy=SimpleNamespace(write_effective_config_files=write, build_items=build)
    )
    args = SimpleNamespace(
        levels=["l1"],
        traj_indices=[0],
        keep_static_support_contact=True,
        static_support_clearance_m=0.001,
    )
    options = SimpleNamespace(upstream=upstream, output=output, speed_mps=0.09)
    assert ev._effective_items(source, args, options, [{"item_id": "selected"}]) == [
        {"item_id": "selected"}
    ]
    assert original.read_bytes() == before


class FakePolicy:
    def __init__(self, checkpoint, fail=False):
        self.identity = identity(checkpoint)
        self.records = []
        self.calls = []
        self.fail = fail
        self.chunk_steps = 8

    def prepare(self):
        self.calls.append(("prepare", threading.get_ident()))
        return self.identity

    def reset(self):
        self.calls.append(("reset", threading.get_ident()))

    def get_action_chunk(self, observation, language):
        self.calls.append(("infer", threading.get_ident()))
        self.records.append(
            {
                "request_sequence": len(self.records),
                "status": "complete",
                "server_audit": {"graph_execution": {"cold_capture": False}},
            }
        )
        if self.fail:
            raise ValueError("failed warmup")
        return np.zeros((8, 7))

    def close(self):
        self.calls.append(("close", threading.get_ident()))


def fake_episode_inputs(tmp_path, *, fail=False):
    class Base:
        def __init__(self, item, args, worker_slot=0):
            self.item, self.args = item, args
            self.preview_elapsed = 2.0
            self.preview_dt = 0.05
            self.preview_states = []
            self.use_preview = True

        def _setup_world(self, env, init):
            return (
                {"agentview_image": np.zeros((2, 2, 3), dtype=np.uint8)},
                80,
                10,
                40,
                [],
            )

        def _build_controller(self, policy, language):
            controller = SimpleNamespace(
                executor=ThreadPoolExecutor(max_workers=1),
                policy=policy,
                future=None,
                latency_sampler=SimpleNamespace(fixed_seconds=0, source="fixed"),
                reset=policy.reset,
                snapshot_request_observation=lambda packet, actions: dict(packet.obs),
                tick=lambda packet, actions: [],
            )
            return controller

        def _apply_sync_latency(self, env, duration, *, on_substep=None):
            for index in range(round(duration / 0.01)):
                env.sim.data.time += 0.01
                if on_substep is not None:
                    on_substep((index + 1) * 0.01)
            self.preview_elapsed += duration
            return duration

    source = SimpleNamespace(
        runtime=SimpleNamespace(
            RealtimeEpisode=Base, _target_entries=lambda item, env: []
        )
    )
    options = SimpleNamespace(
        output=tmp_path,
        expected_checkpoint=tmp_path,
        precision="bf16",
        suite="libero_spatial",
        warmup_requests=2,
        calibration_requests=3,
        calibration_max_p95_ratio=1000,
        rpc_timeout_seconds=2,
        timing_rule="paper_quantized_sync",
        calibrate=False,
        save_videos=True,
        video_fps=20,
    )
    session = {"calibration": ev.counterfactual_latency(0.06), "binding": None}
    item = {"item_id": "item0", "suite": "libero_spatial", "level": "l1"}
    args = SimpleNamespace(calibrated_latency_seconds=0)
    policy = FakePolicy(tmp_path, fail=fail)
    episode = ev.make_episode_type(source, options, session)(item, args, policy=policy)
    episode._setup_world(
        SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(time=3.0))), None
    )
    return episode, policy, session


def test_episode_warms_owning_executor_then_reset_before_formal_request(tmp_path):
    episode, policy, session = fake_episode_inputs(tmp_path)
    controller = episode._build_controller(policy, "pick cup")
    try:
        assert controller.latency_sampler.fixed_seconds == 0.1
        assert session["calibration"]["raw_latency_s"] == 0.06
        assert [record["phase"] for record in policy.records] == ["warmup", "warmup"]
        controller.reset()
        assert [name for name, _ in policy.calls] == [
            "prepare",
            "infer",
            "infer",
            "reset",
            "prepare",
        ]
        assert policy.calls[-2][1] == threading.get_ident()
        owners = {owner for name, owner in policy.calls if name != "reset"}
        assert len(owners) == 1 and threading.get_ident() not in owners
        packet = SimpleNamespace(
            step_idx=8, sim_time_s=0.46, obs=episode.initial_observation
        )
        capture = controller.snapshot_request_observation(packet, None)[
            "__pir2_capture"
        ]
        assert capture["source_tick_kind"] == "action_step"
        assert capture["sim_time_s"] == 0.46
    finally:
        controller.close()
    assert policy.calls[-1][0] == "close"


def test_episode_warmup_failure_closes_executor_and_policy(tmp_path):
    episode, policy, _ = fake_episode_inputs(tmp_path, fail=True)
    with pytest.raises(ValueError, match="failed warmup"):
        episode._build_controller(policy, "pick cup")
    assert policy.calls[-1][0] == "close"
    assert policy.records[0]["phase"] == "warmup"


def test_failure_artifacts_retain_nonfinite_model_outputs_in_strict_json(tmp_path):
    path = tmp_path / "failure-audit.json"
    ev._write_json(path, {"actions": np.array([[float("nan"), float("inf")]])})
    output = json.loads(
        path.read_text(), parse_constant=lambda token: pytest.fail(token)
    )
    assert output["actions"] == [[{"nonfinite": "nan"}, {"nonfinite": "inf"}]]


def test_latency_advances_once_and_video_map_includes_joint_held_motion(tmp_path):
    episode, policy, _ = fake_episode_inputs(tmp_path)
    controller = episode._build_controller(policy, "pick cup")
    try:
        controller.reset()
        observed_substeps = []
        advanced = episode._apply_sync_latency(
            episode.env, 0.1, on_substep=observed_substeps.append
        )
        assert advanced == 0.1
        assert episode.rollout_sim_time_s == 0.1
        assert observed_substeps[-1] == 0.1
        assert [row["sample_sim_time_s"] for row in episode.video_samples] == [
            0.05,
            0.1,
        ]
        assert [row["media_time_s"] for row in episode.video_samples] == [0.0, 0.05]
        assert [row["phase"] for row in episode.video_samples] == ["inference_hold"] * 2
        assert [row["kind"] for row in episode.motion_events][-2:] == [
            "hold_started",
            "hold_completed",
        ]
    finally:
        controller.close()
