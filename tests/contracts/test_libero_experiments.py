"""Deployment planning and paired fixtures; no model or GPU execution."""

import importlib
from copy import deepcopy
from pathlib import Path

import pytest


def module():
    return importlib.import_module("coexecution.libero.experiments")


@pytest.fixture
def plan():
    return {
        "runtime_root": "/runtime",
        "model_python": "/model-python",
        "evaluation_python": "/eval-python",
        "upstream": "/fixed-upstream",
        "libero_config": "/libero-config",
        "gpus": {"action": 0, "vlm": 0, "simulation": 0},
        "ports": {"action": 5680, "vlm": 5682},
        "episodes_per_task": 1,
        "max_steps": 720,
        "seed": 1000,
        "protocols": ["deployment"],
        "conditions": [
            {
                "name": "bf16-baseline",
                "checkpoint": "/weights",
                "variant": "flow",
                "flags": [],
            },
            {
                "name": "bf16-common",
                "checkpoint": "/weights",
                "variant": "flow",
                "flags": ["--input-preparation", "--static-vlm", "--vlm-cuda-graph"],
            },
        ],
    }


def test_commands_use_declared_ports_full_suite_and_separate_workers(plan):
    m = module()
    m.validate_plan(plan)
    commands = m.build_commands(
        plan, plan["conditions"][1], "deployment", Path("/output")
    )
    assert commands["action"][0] == "/model-python"
    assert commands["vlm"][0] == "/model-python"
    for role, port in (("action", "5680"), ("vlm", "5682")):
        command = commands[role]
        assert command[command.index("--role") + 1] == role
        assert command[command.index("--port") + 1] == port
        assert command[command.index("--device") + 1] == "cuda:0"
        assert "--vlm-device" not in command
        assert "--vlm-cuda-graph" in command
    evaluate = commands["evaluate"]
    assert evaluate[0] == "/eval-python"
    assert evaluate[evaluate.index("--control-hz") + 1] == "20"
    assert evaluate[evaluate.index("--port") + 1] == "5680"
    assert evaluate[evaluate.index("--slow-port") + 1] == "5682"
    assert evaluate[evaluate.index("--task-ids") + 1 :] == [str(i) for i in range(10)]


@pytest.mark.parametrize("layout", ["split", "paired"])
def test_image_resolution_plan_default_override_and_explicit_disable(plan, layout):
    plan["server_layout"] = layout
    plan["image_resolution"] = 192
    plan["conditions"][1]["image_resolution"] = 128
    module().validate_plan(plan)
    for case, size in zip(plan["conditions"], (192, 128), strict=True):
        commands = module().build_commands(plan, case, "deployment", Path("/out"))
        roles = ("paired",) if layout == "paired" else ("action", "vlm")
        for role in roles:
            assert commands[role][
                commands[role].index("--image-resolution") + 1
            ] == str(size)
        assert "--image-resolution" not in commands["evaluate"]
    plan["conditions"][1]["image_resolution"] = None
    commands = module().build_commands(
        plan, plan["conditions"][1], "deployment", Path("/out")
    )
    assert all("--image-resolution" not in command for command in commands.values())


@pytest.mark.parametrize("coupling", ["synchronous", "one_step_async", "chunked_async"])
def test_single_policy_coupling_keeps_resolution_and_runtime_receipt(plan, coupling):
    plan.update(
        execution_coupling=coupling,
        execution_clock="ready",
        execution_prefix_steps=1,
        image_resolution=192,
        runtime_receipt=True,
    )
    if coupling == "chunked_async":
        plan["async_overlap_steps"] = 1
    module().validate_plan(plan)
    commands = module().build_commands(
        plan, plan["conditions"][0], "deployment", Path("/out")
    )
    assert set(commands) == {"action", "evaluate"}
    action = commands["action"]
    assert action[action.index("--image-resolution") + 1] == "192"
    assert "--runtime-receipt" in action
    assert "--image-resolution" not in commands["evaluate"]
    assert "--runtime-receipt" not in commands["evaluate"]


@pytest.mark.parametrize("size", [False, 0, -32, 128.0, "128"])
@pytest.mark.parametrize("location", ["plan", "case"])
def test_invalid_image_resolution_fails_before_worker_launch(plan, size, location):
    target = plan if location == "plan" else plan["conditions"][0]
    target["image_resolution"] = size
    with pytest.raises(ValueError, match="Image resolution"):
        module().validate_plan(plan)


def test_execution_clock_plan_default_and_case_override(plan):
    plan["execution_clock"] = "ready"
    plan["conditions"][1]["execution_clock"] = "paced"
    module().validate_plan(plan)
    for case, expected in zip(plan["conditions"], ("ready", "paced"), strict=True):
        command = module().build_commands(plan, case, "deployment", Path("/out"))[
            "evaluate"
        ]
        assert "--execution-clock" in command
        assert command[command.index("--execution-clock") + 1] == expected


def test_runtime_receipt_is_explicit_and_reaches_only_model_workers(plan):
    plan["runtime_receipt"] = True
    plan["conditions"][1]["runtime_receipt"] = False
    module().validate_plan(plan)
    for case, enabled in zip(plan["conditions"], (True, False), strict=True):
        commands = module().build_commands(plan, case, "deployment", Path("/out"))
        assert ("--runtime-receipt" in commands["action"]) is enabled
        assert ("--runtime-receipt" in commands["vlm"]) is enabled
        assert "--runtime-receipt" not in commands["evaluate"]
    plan["runtime_receipt"] = "yes"
    with pytest.raises(ValueError, match="runtime_receipt"):
        module().validate_plan(plan)


@pytest.mark.parametrize("change", ["variant", "protocol", "fixed", "unknown"])
def test_ready_plan_rejects_incompatible_settings_before_launch(plan, change):
    plan["execution_clock"] = "ready"
    if change == "variant":
        plan["conditions"][0]["variant"] = "pir2"
    elif change == "protocol":
        plan["protocols"] = ["algorithm", "deployment"]
    elif change == "fixed":
        plan["conditions"][0]["deployment_fixed_delay"] = 1
    else:
        plan["conditions"][0]["execution_clock"] = "unpaced"
    with pytest.raises(ValueError, match="Ready|ready"):
        module().validate_plan(plan)


@pytest.mark.parametrize("transport", ["numpy_bf16", "cuda_local"])
def test_paired_layout_has_one_owned_process_and_two_explicit_endpoints(
    plan, transport
):
    plan.update(server_layout="paired", feature_transport=transport)
    module().validate_plan(plan)
    commands = module().build_commands(
        plan, plan["conditions"][0], "deployment", Path("/output")
    )
    assert set(commands) == {"paired", "evaluate"}
    command = commands["paired"]
    assert command[command.index("--server-layout") + 1] == "paired"
    assert command[command.index("--vlm-device") + 1] == "cuda:0"
    assert command[command.index("--vlm-port") + 1] == "5682"
    assert (
        commands["evaluate"][commands["evaluate"].index("--server-layout") + 1]
        == "paired"
    )


@pytest.mark.parametrize("change", ["layout", "placement", "variant"])
def test_resident_plan_rejects_incompatible_placements_before_launch(plan, change):
    plan.update(server_layout="paired", feature_transport="cuda_local")
    if change == "layout":
        plan["server_layout"] = "split"
    elif change == "placement":
        plan["gpus"]["vlm"] = 1
    else:
        plan["conditions"][0]["variant"] = "pir2"
    with pytest.raises(ValueError):
        module().validate_plan(plan)


@pytest.mark.parametrize("case_rate, expected", [(None, 15), (12, 12), (0, 0)])
def test_vision_rate_plan_default_and_case_override_reach_only_deployment(
    plan, case_rate, expected
):
    m = module()
    plan["vision_max_hz"] = 15
    plan["protocols"] = ["algorithm", "deployment"]
    case = plan["conditions"][0]
    if case_rate is not None:
        case["vision_max_hz"] = case_rate
    m.validate_plan(plan)
    command = m.build_commands(plan, case, "deployment", Path("/output"))["evaluate"]
    assert "--vision-max-hz" in command
    assert float(command[command.index("--vision-max-hz") + 1]) == expected
    assert command[command.index("--control-hz") + 1] == "20"
    algorithm = m.build_commands(plan, case, "algorithm", Path("/output"))["evaluate"]
    assert "--vision-max-hz" not in algorithm
    for role in ("action", "vlm"):
        assert (
            "--vision-max-hz"
            not in m.build_commands(plan, case, "deployment", Path("/output"))[role]
        )


@pytest.mark.parametrize("location", ["plan", "case"])
@pytest.mark.parametrize(
    "rate", [-1, float("nan"), float("inf"), -float("inf"), None, "15", True]
)
def test_vision_rate_validation_precedes_launch(plan, location, rate):
    target = plan if location == "plan" else plan["conditions"][0]
    target["vision_max_hz"] = rate
    with pytest.raises(ValueError, match="finite.*nonnegative"):
        module().validate_plan(plan)


@pytest.mark.parametrize("location", ["plan", "case"])
def test_nonzero_vision_rate_requires_a_deployment_protocol(plan, location):
    plan["protocols"] = ["algorithm"]
    target = plan if location == "plan" else plan["conditions"][0]
    target["vision_max_hz"] = 15
    with pytest.raises(ValueError, match="deployment"):
        module().validate_plan(plan)


def test_omitted_vision_rate_preserves_existing_evaluator_command(plan):
    commands = module().build_commands(
        plan, plan["conditions"][0], "deployment", Path("/output")
    )
    assert "--vision-max-hz" not in commands["evaluate"]


def test_feature_controls_reach_workers_and_deployment_without_weight_flags(plan):
    plan["feature_transport"] = "numpy_bf16"
    plan["feature_rpc"] = "combined"
    plan["protocols"] = ["algorithm", "deployment"]
    case = plan["conditions"][0]
    module().validate_plan(plan)
    commands = module().build_commands(plan, case, "deployment", Path("/output"))
    for role in ("action", "vlm", "evaluate"):
        command = commands[role]
        assert command[command.index("--feature-transport") + 1] == "numpy_bf16"
    assert (
        commands["evaluate"][commands["evaluate"].index("--feature-rpc") + 1]
        == "combined"
    )
    for role in ("action", "vlm"):
        assert "--feature-rpc" not in commands[role]
    case["feature_transport"] = "numpy_fp32"
    case["feature_rpc"] = "separate"
    command = module().build_commands(plan, case, "deployment", Path("/output"))[
        "evaluate"
    ]
    assert command[command.index("--feature-transport") + 1] == "numpy_fp32"
    assert command[command.index("--feature-rpc") + 1] == "separate"
    algorithm = module().build_commands(plan, case, "algorithm", Path("/output"))[
        "evaluate"
    ]
    assert "--feature-rpc" not in algorithm
    assert "--feature-transport" not in algorithm


@pytest.mark.parametrize(
    "name, value", [("feature_transport", "float16"), ("feature_rpc", "other")]
)
@pytest.mark.parametrize("location", ["plan", "case"])
def test_invalid_feature_options_rejected_before_launch(plan, name, value, location):
    target = plan if location == "plan" else plan["conditions"][0]
    target[name] = value
    with pytest.raises(ValueError, match="feature"):
        module().validate_plan(plan)


def test_feature_controls_require_deployment_in_experiment_protocols(plan):
    plan["protocols"] = ["algorithm"]
    plan["feature_rpc"] = "combined"
    with pytest.raises(ValueError, match="deployment"):
        module().validate_plan(plan)


@pytest.mark.parametrize("change", ["ports", "names", "flags", "episodes", "protocol"])
def test_invalid_plan_is_rejected_before_starting_any_process(plan, change):
    if change == "ports":
        plan["ports"]["vlm"] = plan["ports"]["action"]
    elif change == "names":
        plan["conditions"][1]["name"] = plan["conditions"][0]["name"]
    elif change == "flags":
        plan["conditions"][0]["flags"] = ["--checkpoint", "/other"]
    elif change == "episodes":
        plan["episodes_per_task"] = 0
    else:
        plan["protocols"] = ["unknown"]
    with pytest.raises(ValueError):
        module().validate_plan(plan)


def test_full_paired_fixtures_require_every_episode_and_matching_hashes():
    m = module()
    rows = [
        {"task_id": task, "episode_id": 0, **dict.fromkeys(m.FIXTURE_FIELDS, str(task))}
        for task in range(10)
    ]
    candidates = {"baseline": rows, "common": deepcopy(rows)}
    assert m.paired_fixtures(candidates, episodes_per_task=1)["passed"]
    candidates["common"].pop()
    assert not m.paired_fixtures(candidates, episodes_per_task=1)["passed"]
    candidates["common"] = deepcopy(rows)
    candidates["common"][3]["initial_rgb_sha256"] = "different"
    assert not m.paired_fixtures(candidates, episodes_per_task=1)["passed"]


def test_old_ready_marker_cannot_admit_a_new_worker(plan, tmp_path, monkeypatch):
    m = module()
    (tmp_path / "run.log").write_text("TIMED_PROTOCOL_SERVER_READY old worker\n")

    class Process:
        pid = 123456

        def poll(self):
            return None

    monkeypatch.setattr(m.subprocess, "Popen", lambda *args, **kwargs: Process())
    owned = m.OwnedProcesses(plan)
    owned.start("action", ["worker"], tmp_path, "action")
    with pytest.raises(TimeoutError):
        owned.ready("action", tmp_path, timeout=0)
    assert "old worker" in (tmp_path / "run.log").read_text()


def test_paired_readiness_requires_both_owned_endpoint_markers(plan, tmp_path):
    from types import SimpleNamespace

    owned = module().OwnedProcesses(plan)
    owned.children["paired"] = SimpleNamespace(poll=lambda: None)
    owned.logs["paired"] = tmp_path / "paired.log"
    owned.logs["paired"].write_text("TIMED_PROTOCOL_SERVER_READY action\n")
    with pytest.raises(TimeoutError):
        owned.ready("paired", tmp_path, timeout=0, endpoints=2)
    owned.logs["paired"].write_text(
        "TIMED_PROTOCOL_SERVER_READY action\nTIMED_PROTOCOL_SERVER_READY vlm\n"
    )
    owned.ready("paired", tmp_path, timeout=0, endpoints=2)


def test_cleanup_signals_each_owned_pid_once_even_with_endpoint_aliases(
    plan, monkeypatch
):
    from types import SimpleNamespace

    m = module()
    signals, waits = [], []
    owned = m.OwnedProcesses(plan)
    process = SimpleNamespace(
        pid=12345, poll=lambda: None, wait=lambda **kwargs: waits.append(kwargs)
    )
    owned.children = {"action": process, "vlm": process}
    monkeypatch.setattr(m.os, "killpg", lambda *args: signals.append(args))
    owned.close()
    assert len(signals) == len(waits) == 1


def test_failed_child_cleanup_still_stops_other_owned_workers(plan, monkeypatch):
    m = module()
    signals = []

    class Process:
        def __init__(self, pid, blocked):
            self.pid, self.blocked, self.waited = pid, blocked, False

        def poll(self):
            return None

        def wait(self, timeout):
            self.waited = True
            if self.blocked:
                raise m.subprocess.TimeoutExpired("worker", timeout)

    first, second = Process(123450, False), Process(123451, True)
    owned = m.OwnedProcesses(plan)
    owned.children = {"first": first, "second": second}
    monkeypatch.setattr(m.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(RuntimeError, match="stop"):
        owned.close()
    assert first.waited and second.waited
    assert {pid for pid, _ in signals} == {first.pid, second.pid}


def test_missing_launch_paths_reject_before_gpu_admission_or_processes(plan):
    with pytest.raises(ValueError, match="path"):
        module().validate_launch_paths(plan)


def test_official_predicted_and_native_decoded_horizons_are_distinct():
    from types import SimpleNamespace

    from coexecution.libero import service

    core = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(streaming=False, action_horizon=40)
        ),
        modality_configs={"action": SimpleNamespace(delta_indices=list(range(16)))},
    )
    assert service.policy_horizons(core, "flow") == {
        "predicted_horizon": 40,
        "decoded_horizon": 16,
    }
    core.model.config.streaming = True
    core.modality_configs["action"].delta_indices = list(range(40))
    assert service.policy_horizons(core, "pir2")["decoded_horizon"] == 40
    core.modality_configs["action"].delta_indices = list(range(4))
    with pytest.raises(ValueError):
        service.policy_horizons(core, "pir2")


@pytest.mark.parametrize(
    "invalid", ["keys", "missing_age", "nonfinite_age", "negative_age"]
)
def test_deployment_completion_requires_exact_episode_keys_and_actual_ages(
    plan, tmp_path, monkeypatch, invalid
):
    from coexecution import feature_usage, queue_report

    condition = {
        "episodes": [
            {
                "metadata": {"task_id": task, "episode_id": 0},
                "ticks": [
                    {
                        "fallback": False,
                        "executed_image_age_ms": 10.0,
                        "executed_state_age_ms": 12.0,
                    }
                ],
            }
            for task in range(10)
        ]
    }
    if invalid == "keys":
        condition["episodes"][0]["metadata"]["task_id"] = 10
    elif invalid == "missing_age":
        condition["episodes"][0]["ticks"][0]["executed_image_age_ms"] = None
    elif invalid == "nonfinite_age":
        condition["episodes"][0]["ticks"][0]["executed_state_age_ms"] = float("nan")
    else:
        condition["episodes"][0]["ticks"][0]["executed_state_age_ms"] = -1
    monkeypatch.setattr(queue_report, "condition_report", lambda path: condition)
    monkeypatch.setattr(
        feature_usage,
        "analyze_report",
        lambda *args, **kwargs: {
            "conditions": [
                {
                    "read_accounting_complete": True,
                    "execution_accounting_complete": True,
                }
            ]
        },
    )
    with pytest.raises(RuntimeError):
        module().deployment_report(plan, tmp_path)


def test_lookup_race_cleanup_timeout_still_visits_other_workers(plan, monkeypatch):
    m = module()
    visited = []

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return None

        def wait(self, timeout):
            visited.append(self.pid)
            if self.pid == 123451:
                raise m.subprocess.TimeoutExpired("worker", timeout)

    owned = m.OwnedProcesses(plan)
    owned.children = {"action": Process(123450), "vlm": Process(123451)}

    def kill(pid, sig):
        if pid == 123451:
            raise ProcessLookupError("worker just exited")

    monkeypatch.setattr(m.os, "killpg", kill)
    with pytest.raises(RuntimeError, match="stop"):
        owned.close()
    assert 123450 in visited


def test_admission_rechecks_transient_activity_before_binding_ports(plan, monkeypatch):
    m = module()
    activity = iter(["126, 30", "126, 0"])
    calls, waits, ports = [], [], []

    def query(command, **kwargs):
        calls.append(command)
        return (
            next(activity)
            if "--query-gpu=memory.used,utilization.gpu" in command
            else ""
        )

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def bind(self, address):
            assert len(calls) == 4
            ports.append(address[1])

    monkeypatch.setattr(m.subprocess, "check_output", query)
    monkeypatch.setattr(m.time, "sleep", waits.append)
    monkeypatch.setattr(m.socket, "socket", Socket)
    m._admit_devices(plan)
    assert waits and sum(waits) <= 2
    assert ports == list(plan["ports"].values())


def test_admission_persistent_activity_is_bounded_and_explains_rejection(
    plan, monkeypatch
):
    m = module()
    waits = []
    monkeypatch.setattr(
        m.subprocess,
        "check_output",
        lambda command, **kwargs: (
            "126, 30" if "--query-gpu=memory.used,utilization.gpu" in command else ""
        ),
    )
    monkeypatch.setattr(m.time, "sleep", waits.append)
    monkeypatch.setattr(m.socket, "socket", lambda: pytest.fail("Busy GPU admitted"))
    with pytest.raises(RuntimeError, match="memory=126.*utilization=30"):
        m._admit_devices(plan)
    assert waits and sum(waits) <= 2


def test_admission_never_retries_a_visible_compute_process(plan, monkeypatch):
    m = module()
    monkeypatch.setattr(
        m.subprocess,
        "check_output",
        lambda command, **kwargs: (
            "126, 0"
            if "--query-gpu=memory.used,utilization.gpu" in command
            else "12345"
        ),
    )
    monkeypatch.setattr(
        m.time, "sleep", lambda seconds: pytest.fail("Active process waited")
    )
    with pytest.raises(RuntimeError, match="12345"):
        m._admit_devices(plan)


def test_failed_admission_publishes_current_failure_instead_of_running(
    plan, tmp_path, monkeypatch
):
    m = module()
    monkeypatch.setattr(m, "validate_launch_paths", lambda plan: None)

    def reject(plan):
        raise RuntimeError("GPU admission rejected")

    monkeypatch.setattr(m, "_admit_devices", reject)
    with pytest.raises(RuntimeError, match="admission rejected"):
        m.run(plan, tmp_path)
    import json

    status = json.loads((tmp_path / "status.json").read_text())
    assert status["phase"] == "failed"
    assert status["condition"] == "bf16-baseline"


def test_selected_debug_tasks_record_real_replay_and_keep_fixture_scope(plan):
    plan.update(task_ids=[0, 3], record_replay=True, replay_max_frames=721)
    module().validate_plan(plan)
    command = module().build_commands(
        plan, plan["conditions"][0], "deployment", Path("/output")
    )["evaluate"]
    assert "--record-replay" in command
    assert command[command.index("--replay-max-frames") + 1] == "721"
    assert command[command.index("--task-ids") + 1 :] == ["0", "3"]
    rows = [
        {
            "task_id": task,
            "episode_id": 0,
            **{field: "sha" for field in module().FIXTURE_FIELDS},
        }
        for task in (0, 3)
    ]
    assert module().paired_fixtures(
        {"bf16": rows, "w8a8": rows}, episodes_per_task=1, task_ids=[0, 3]
    )["passed"]
    assert not module().paired_fixtures({"bf16": rows}, episodes_per_task=1)["passed"]


@pytest.mark.parametrize("ids", [[], [0, 0], [10], [True], ["0"]])
def test_invalid_selected_tasks_rejected(plan, ids):
    plan["task_ids"] = ids
    with pytest.raises(ValueError, match="task"):
        module().validate_plan(plan)


@pytest.mark.parametrize(
    "selected", [None, "libero_spatial", "libero_object", "libero_goal", "libero_10"]
)
@pytest.mark.parametrize("full_suite", [False, True], ids=["subset", "full"])
def test_deployment_report_reads_actual_trace_and_reports_suite_scope(
    plan, tmp_path, selected, full_suite
):
    import json

    tasks = list(range(10)) if full_suite else [0]
    plan.update(task_ids=tasks, conditions=plan["conditions"][:1])
    if selected is not None:
        plan["suite"] = selected
    directory = tmp_path / plan["conditions"][0]["name"] / "deployment"
    directory.mkdir(parents=True)
    metadata = {
        "task_id": 0,
        "episode_id": 0,
        "seed": 1000,
        "success": True,
        "control_rate_valid": True,
    }
    (directory / "episodes.jsonl").write_text(
        "\n".join(json.dumps({**metadata, "task_id": task}) for task in tasks) + "\n"
    )
    events = [
        {
            "kind": "control_start",
            "t_s": 10,
            "period_s": 0.05,
            "feature_sequence": 0,
            "feature_source_tick": 0,
            "feature_capture_s": 9.9,
            "initial_capture_s": 9.9,
        },
        {
            "kind": "control_tick",
            "t_s": 10.001,
            "tick": 0,
            "fallback": False,
            "executed_slot": {"producer_kind": "bootstrap"},
        },
        {"kind": "control_end", "t_s": 10.05, "control_steps": 1},
    ]
    for task in tasks:
        (directory / f"task{task}-episode0-trace.jsonl").write_text(
            "\n".join(map(json.dumps, events))
        )
    report = module().deployment_report(plan, tmp_path)
    expected_name = {
        None: "Spatial",
        "libero_spatial": "Spatial",
        "libero_object": "Object",
        "libero_goal": "Goal",
        "libero_10": "Long",
    }[selected]
    expected_scope = (
        f"Full {expected_name} suite" if full_suite else f"{expected_name} task subset"
    )
    assert (
        report["scope"]
        == expected_scope
        + ", paired declared initial states; actual executed image/state ages"
    )
    assert report["task_ids"] == tasks
    assert len(report["conditions"][0]["episodes"]) == len(tasks)
    assert (tmp_path / "feature-usage.json").is_file()


def test_synchronous_plan_owns_only_one_worker_and_propagates_coupling(plan):
    plan.update(execution_coupling="synchronous", synchronous_chunk_steps=8)
    module().validate_plan(plan)
    commands = module().build_commands(
        plan, plan["conditions"][0], "deployment", Path("/out")
    )
    assert set(commands) == {"action", "evaluate"}
    evaluate = commands["evaluate"]
    assert evaluate[evaluate.index("--execution-coupling") + 1] == "synchronous"
    assert evaluate[evaluate.index("--synchronous-chunk-steps") + 1] == "8"
    plan["conditions"][1]["server_layout"] = "paired"
    with pytest.raises(ValueError, match="Synchronous"):
        module().validate_plan(plan)


def test_declared_uuid_mask_wins_over_physical_ordinal(plan):
    uuid = "GPU-a96bcf12-94d7-95fc-c078-b597412cdd63"
    plan["gpu_uuids"] = {role: uuid for role in plan["gpus"]}
    module().validate_plan(plan)
    assert module().environment(plan, "action")["CUDA_VISIBLE_DEVICES"] == uuid
    assert module().environment(plan, "simulation")["MUJOCO_EGL_DEVICE_ID"] == "0"
    for wrong in [{"action": uuid}, dict.fromkeys(plan["gpus"], "GPU-a96b")]:
        plan["gpu_uuids"] = wrong
        with pytest.raises(ValueError, match="UUID"):
            module().validate_plan(plan)


def test_uuid_index_mismatch_rejects_before_resource_admission(plan, monkeypatch):
    m = module()
    declared = "GPU-a96bcf12-94d7-95fc-c078-b597412cdd63"
    observed = "GPU-4a56586b-2b20-8158-2085-c97765614e2c"
    plan["gpu_uuids"] = dict.fromkeys(plan["gpus"], declared)
    calls = []

    def query(command, **kwargs):
        calls.append(command)
        assert "--query-gpu=index,uuid" in command
        return f"0, {observed}\n"

    monkeypatch.setattr(m.subprocess, "check_output", query)
    with pytest.raises(RuntimeError, match="UUID.*mismatch"):
        m._admit_devices(plan)
    assert len(calls) == 1


def test_uuid_admission_returns_verified_role_mapping(plan, monkeypatch):
    m = module()
    uuid = "GPU-a96bcf12-94d7-95fc-c078-b597412cdd63"
    plan["gpu_uuids"] = dict.fromkeys(plan["gpus"], uuid)

    def query(command, **kwargs):
        if "--query-gpu=index,uuid" in command:
            return f"0, {uuid}\n"
        if "--query-gpu=memory.used,utilization.gpu" in command:
            return "0, 0"
        return ""

    monkeypatch.setattr(m.subprocess, "check_output", query)
    mapping = m._admit_devices(plan)
    assert mapping == {
        role: {"physical_index": 0, "uuid": uuid} for role in plan["gpus"]
    }


def test_synchronous_warmup_timeout_reaches_evaluator(plan):
    plan.update(
        execution_coupling="synchronous", synchronous_warmup_timeout_seconds=600
    )
    m = module()
    m.validate_plan(plan)
    command = m.build_commands(plan, plan["conditions"][0], "deployment", Path("/out"))[
        "evaluate"
    ]
    assert command[command.index("--synchronous-warmup-timeout-seconds") + 1] == "600"
    plan["conditions"][0]["synchronous_warmup_timeout_seconds"] = -1
    with pytest.raises(ValueError, match="warmup.*timeout"):
        m.validate_plan(plan)


def test_one_step_async_plan_owns_single_worker_and_prefix_alias(plan):
    plan.update(
        execution_coupling="one_step_async",
        execution_clock="ready",
        execution_prefix_steps=1,
    )
    module().validate_plan(plan)
    commands = module().build_commands(
        plan, plan["conditions"][0], "deployment", Path("/out")
    )
    assert set(commands) == {"action", "evaluate"}
    evaluate = commands["evaluate"]
    assert evaluate[evaluate.index("--execution-coupling") + 1] == "one_step_async"
    assert evaluate[evaluate.index("--synchronous-chunk-steps") + 1] == "1"
    plan["conditions"][1]["synchronous_chunk_steps"] = 8
    with pytest.raises(ValueError, match="exactly one"):
        module().validate_plan(plan)
    plan["conditions"][1]["execution_prefix_steps"] = 1
    with pytest.raises(ValueError, match="Conflicting"):
        module().validate_plan(plan)


def launchable_plan(plan, tmp_path, horizon=16):
    """Real checkpoint JSON layout and files, without a model or CUDA runtime."""
    import json

    root = tmp_path / "runtime"
    upstream = root / "upstream/learning/Isaac-GR00T"
    checkpoint = tmp_path / "checkpoint"
    for path in (
        root / "coexecution/libero/service.py",
        root / "scripts/evaluate_libero_protocol.py",
        tmp_path / "model-python",
        tmp_path / "evaluation-python",
        tmp_path / "libero-config",
        upstream / "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
        upstream / "gr00t/policy/decoupled_policy.py",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"action_horizon": 40}))
    (checkpoint / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "Gr00tN1d7Processor",
                "processor_kwargs": {
                    "modality_configs": {
                        "libero_sim": {
                            "action": {"delta_indices": list(range(horizon))}
                        }
                    }
                },
            }
        )
    )
    result = deepcopy(plan)
    result.update(
        runtime_root=str(root),
        model_python=str(tmp_path / "model-python"),
        evaluation_python=str(tmp_path / "evaluation-python"),
        upstream=str(upstream),
        libero_config=str(tmp_path / "libero-config"),
        execution_clock="ready",
        execution_coupling="chunked_async",
        execution_prefix_steps=8,
        async_overlap_steps=1,
    )
    for condition in result["conditions"]:
        condition["checkpoint"] = str(checkpoint)
    return result


@pytest.mark.parametrize("coupling", ["synchronous", "chunked_async"])
@pytest.mark.parametrize("override", [False, True], ids=["global", "last-condition"])
def test_native_prefix_preflight_rejects_before_any_gpu_or_child(
    plan, tmp_path, monkeypatch, coupling, override
):
    m = module()
    plan = launchable_plan(plan, tmp_path)
    plan["execution_coupling"] = coupling
    plan["async_overlap_steps"] = 1 if coupling == "chunked_async" else 0
    if override:
        plan["conditions"][-1]["execution_prefix_steps"] = 17
    else:
        plan["execution_prefix_steps"] = 17
    activity = []
    monkeypatch.setattr(m, "_admit_devices", lambda *_: activity.append("gpu"))
    monkeypatch.setattr(m.OwnedProcesses, "start", lambda *_: activity.append("child"))
    with pytest.raises(ValueError, match="prefix 17 exceeds native decoded horizon 16"):
        m.run(plan, tmp_path / "output")
    assert activity == []
    assert not (tmp_path / "output").exists()


def test_native_prefix_preflight_uses_condition_override_and_actual_length(
    plan, tmp_path
):
    m = module()
    plan = launchable_plan(plan, tmp_path, horizon=24)
    plan["execution_prefix_steps"] = 30
    for condition in plan["conditions"]:
        condition["execution_prefix_steps"] = 24
    m.validate_plan(plan)
    m.validate_launch_paths(plan)
    plan["conditions"][-1]["execution_prefix_steps"] = 25
    with pytest.raises(ValueError, match="prefix 25 exceeds native decoded horizon 24"):
        m.validate_launch_paths(plan)


def test_one_step_preflight_requires_readable_native_processor(plan, tmp_path):
    m = module()
    plan = launchable_plan(plan, tmp_path)
    plan.update(execution_coupling="one_step_async", execution_prefix_steps=1)
    m.validate_plan(plan)
    m.validate_launch_paths(plan)
    (Path(plan["conditions"][0]["checkpoint"]) / "processor_config.json").unlink()
    with pytest.raises(ValueError, match="decoded action horizon"):
        m.validate_launch_paths(plan)


def test_responsive_plan_global_and_condition_identity(plan):
    plan["deployment_scheduling_policy"] = "responsive"
    for case in plan["conditions"]:
        case["variant"] = "pir2"
        case["flags"] = []
    plan["conditions"][1]["deployment_scheduling_policy"] = "conservative"
    module().validate_plan(plan)
    for case, expected in zip(
        plan["conditions"], ("responsive", "conservative"), strict=True
    ):
        command = module().build_commands(plan, case, "deployment", Path("/out"))[
            "evaluate"
        ]
        assert "--deployment-scheduling-policy" in command
        assert command[command.index("--deployment-scheduling-policy") + 1] == expected
    plan["conditions"][0]["deployment_fixed_delay"] = 1
    with pytest.raises(ValueError, match="Responsive"):
        module().validate_plan(plan)


@pytest.mark.parametrize(
    "suite", ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
)
def test_declared_suite_reaches_static_evaluator(plan, suite):
    plan["suite"] = suite
    m = module()
    m.validate_plan(plan)
    command = m.build_commands(plan, plan["conditions"][0], "deployment", Path("/out"))[
        "evaluate"
    ]
    assert command[command.index("--suite") + 1] == suite


def test_unknown_suite_rejected_before_launch(plan):
    plan["suite"] = "libero_90"
    with pytest.raises(ValueError, match="suite"):
        module().validate_plan(plan)


@pytest.mark.parametrize(
    "selected", [None, "libero_spatial", "libero_object", "libero_goal", "libero_10"]
)
def test_static_main_uses_selected_benchmark_tasks_and_state_files(
    tmp_path, monkeypatch, selected
):
    """Exercise the evaluator loop with distinct suite files; no neural computation."""
    import hashlib
    import json
    import sys
    from types import SimpleNamespace

    import numpy as np

    from scripts import evaluate_libero_protocol as evaluator

    expected_suite = selected or "libero_spatial"
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    opened = []
    initialized = []
    environments = []
    for number, name in enumerate(suites):
        directory = tmp_path / name
        directory.mkdir()
        for task_id in (1, 7):
            np.save(
                directory / f"task{task_id}.npy",
                [[number, task_id, 0], [number, task_id, 1]],
            )
            (directory / f"task{task_id}.bddl").write_text(f"{name}:task{task_id}")

    class Suite:
        def __init__(self, name):
            self.name = name
            opened.append(("suite", name))

        def get_task(self, task_id):
            return SimpleNamespace(
                name=f"{self.name}:task{task_id}",
                language=f"{self.name} language {task_id}",
                problem_folder=self.name,
                bddl_file=f"task{task_id}.bddl",
            )

        def get_task_init_states(self, task_id):
            path = tmp_path / self.name / f"task{task_id}.npy"
            opened.append(("states", path))
            return np.load(path)

    class Environment:
        def __init__(self, bddl, language):
            self.bddl = Path(bddl)
            self.language = language
            self._env = self
            self.settle_actions = []
            self.closed = False
            environments.append(self)

        def seed(self, value):
            self.seed_value = value

        def reset(self):
            return None

        def set_init_state(self, state):
            initialized.append((self.bddl, self.seed_value, state.copy()))
            self.state = state.copy()
            return {"state": self.state}

        def step(self, action):
            self.settle_actions.append(action)
            return {"state": self.state}, 0, False, {}

        def _process_observation(self, raw):
            return raw

        def initial_fingerprints(self, observation):
            return {"fixture_model_sha256": self.bddl.read_text()}

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "libero", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "libero.libero",
        SimpleNamespace(
            benchmark=SimpleNamespace(
                get_benchmark_dict=lambda: {
                    name: lambda name=name: Suite(name) for name in suites
                }
            ),
            get_libero_path=lambda name: str(tmp_path),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "libero_reproducible_env", SimpleNamespace(LiberoEnv=Environment)
    )
    monkeypatch.setitem(
        sys.modules,
        "libero_observations",
        SimpleNamespace(write_video=lambda *_: pytest.fail("unexpected video")),
    )
    monkeypatch.setattr(
        evaluator,
        "wait_server",
        lambda port: SimpleNamespace(
            call_endpoint=lambda *args, **kwargs: {"variant": "flow"}
        ),
    )
    monkeypatch.setattr(
        evaluator, "algorithm_episode", lambda env, obs, *args: ({"success": True}, [])
    )
    original_read_bytes = Path.read_bytes

    def read_bytes(path):
        # Pinned model sources are deliberately absent from public host-only CI.
        if "upstream/learning/Isaac-GR00T/" in path.as_posix():
            return b"host fixture for the pinned upstream source identity"
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    output = tmp_path / "output"
    argv = [
        "evaluate_libero_protocol.py",
        "--protocol",
        "algorithm",
        "--variant",
        "flow",
        "--output",
        str(output),
        "--task-ids",
        "1",
        "7",
        "--episodes-per-task",
        "2",
        "--no-video",
    ]
    if selected is not None:
        argv += ["--suite", selected]
    monkeypatch.setattr(sys, "argv", argv)
    evaluator.main()

    assert opened == [
        ("suite", expected_suite),
        ("states", tmp_path / expected_suite / "task1.npy"),
        ("states", tmp_path / expected_suite / "task7.npy"),
    ]
    expected_states = [
        np.array([suites.index(expected_suite), task_id, episode_id])
        for task_id in (1, 7)
        for episode_id in range(2)
    ]
    for (bddl, seed, state), expected in zip(initialized, expected_states, strict=True):
        task_id, episode_id = expected[1:]
        assert bddl == tmp_path / expected_suite / f"task{task_id}.bddl"
        assert seed == 1000 + 1000 * task_id + episode_id
        np.testing.assert_array_equal(state, expected)
    assert all(env.closed and len(env.settle_actions) == 20 for env in environments)
    assert all(
        action == [0, 0, 0, 0, 0, 0, -1]
        for env in environments
        for action in env.settle_actions
    )
    assert [env.language for env in environments] == [
        f"{expected_suite} language {task_id}" for task_id in (1, 7)
    ]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["config"]["suite"] == expected_suite
    assert summary["config"]["initial_state_indices"] == [0, 1]
    assert summary["complete"] and summary["episodes"] == 4
    assert summary["evaluation_scope"] == "selected_task_subset"
    records = [
        json.loads(line)
        for line in (output / "episodes.jsonl").read_text().splitlines()
    ]
    assert [record["initial_state_sha256"] for record in records] == [
        hashlib.sha256(state.tobytes()).hexdigest() for state in expected_states
    ]


@pytest.mark.parametrize(
    "suite,scope",
    [
        ("libero_spatial", "full_spatial_suite"),
        ("libero_object", "full_object_suite"),
        ("libero_goal", "full_goal_suite"),
        ("libero_10", "full_long_suite"),
    ],
)
def test_static_full_suite_summary_identifies_actual_suite(tmp_path, suite, scope):
    import json

    from scripts.evaluate_libero_protocol import save_summary

    records = [
        {"task_id": task, "episode_id": 0, "task": f"task{task}", "success": True}
        for task in range(10)
    ]
    save_summary(
        tmp_path,
        records,
        {"suite": suite, "episodes_per_task": 1, "protocol": "algorithm"},
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["evaluation_scope"] == scope
    assert summary["expected_episodes"] == 10
    assert summary["complete"] is True
