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


def test_subset_deployment_report_reads_actual_trace_and_reports_selected_scope(
    plan, tmp_path
):
    import json

    plan.update(task_ids=[0], conditions=plan["conditions"][:1])
    directory = tmp_path / plan["conditions"][0]["name"] / "deployment"
    directory.mkdir(parents=True)
    metadata = {
        "task_id": 0,
        "episode_id": 0,
        "seed": 1000,
        "success": True,
        "control_rate_valid": True,
    }
    (directory / "episodes.jsonl").write_text(json.dumps(metadata) + "\n")
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
    (directory / "task0-episode0-trace.jsonl").write_text(
        "\n".join(map(json.dumps, events))
    )
    report = module().deployment_report(plan, tmp_path)
    assert report["task_ids"] == [0]
    assert "subset" in report["scope"] and "Full Spatial" not in report["scope"]
    assert len(report["conditions"][0]["episodes"]) == 1
    assert (tmp_path / "feature-usage.json").is_file()
