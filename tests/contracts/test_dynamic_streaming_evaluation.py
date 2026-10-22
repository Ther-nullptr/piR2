"""Moving-world integration contracts using host physics/RPC doubles only."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.contracts.test_dynamic_streaming import setup as core_setup


class Sim:
    def __init__(self):
        self.data = SimpleNamespace(time=0.0)
        self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
        self.x = 0.0

    def step(self):
        self.data.time += self.model.opt.timestep
        self.x += 0.06 * self.model.opt.timestep

    def render(self, height, width, *, camera_name):
        return np.full((height, width, 3), round(self.x * 1000) % 255, np.uint8)


class Env:
    def __init__(self, **kwargs):
        self.sim = Sim()
        self.env = self
        self.closed = False
        self.actions = []
        self.success_after = 43
        self.fail_after = None
        self.fresh_calls = []
        self.dynamic_fixture_body_state = {"elapsed": 0.0}

    def seed(self, seed):
        self.seed_value = seed

    def _get_observations(self, *, force_update=False):
        self.fresh_calls.append((self.sim.data.time, force_update))
        return {
            "frame": self.sim.data.time,
            "state": {"time": self.sim.data.time},
            "agentview_image": np.zeros((4, 4, 3), np.uint8),
            "robot0_eye_in_hand_image": np.zeros((4, 4, 3), np.uint8),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0, 0, 0, 1.0]),
            "robot0_gripper_qpos": np.zeros(2),
        }

    def check_success(self):
        return len(self.actions) >= self.success_after

    def close(self):
        self.closed = True


class SourceEpisode:
    def __init__(self, item, args, worker_slot):
        self.item, self.args = item, args
        self.preview_elapsed, self.preview_dt = 0.0, 0.05
        self.preview_states = []
        self.use_preview = item.get("preview", True)

    def _configure_env_selection(self):
        pass

    def _setup_world(self, env, init_state):
        for _ in range(40):
            self._step_env(env, np.zeros(7))
        return env._get_observations(), 80, 10, 40, []

    def _step_env(self, env, action, *, on_substep=None, **kwargs):
        env.actions.append(np.array(action, copy=True))
        for index in range(25):
            env.sim.step()
            if on_substep:
                on_substep((index + 1) * 0.002)
            if env.fail_after == len(env.actions):
                raise RuntimeError("physical step interrupted")
        self.preview_elapsed += 0.05
        env.dynamic_fixture_body_state["elapsed"] += 0.05
        return env._get_observations(), 0, False, {}

    def _apply_sync_latency(self, env, duration, *, on_substep=None):
        start = env.sim.data.time
        original = env.sim.model.opt.timestep
        try:
            while env.sim.data.time - start < duration - 1e-12:
                env.sim.model.opt.timestep = min(
                    original, duration - (env.sim.data.time - start)
                )
                env.sim.step()
                if on_substep:
                    on_substep(env.sim.data.time - start)
        finally:
            env.sim.model.opt.timestep = original
        self.preview_elapsed += duration
        return duration


class Writer:
    def __init__(self):
        self.frames = []
        self.closed = False

    def append_data(self, frame):
        self.frames.append(frame.copy())

    def close(self):
        self.closed = True


def fixtures(tmp_path, *, videos=True):
    _, rpc, endpoints, _, _, _ = core_setup(tmp_path)
    # Reset source of the explicit fake policy connection; no neural model.
    options = SimpleNamespace(
        output=tmp_path / "out",
        video_fps=60.0,
        save_videos=videos,
        camera_size=4,
        warmup_requests=1,
        fixed_delay=1,
        max_steps=10,
        seed=0,
        init_state_id=0,
        bootstrap_latency_seconds=0.1,
        s1_latency_seconds=0.05,
        s2_latency_seconds=0.1,
        recovery_latency_seconds=0.3,
        precision="bf16",
        suite="libero_spatial",
        speed_mps=0.06,
        expected_checkpoint=tmp_path / "existing",
        rpc_timeout_seconds=20,
    )
    options.output.mkdir()
    env, writers = Env(), []

    def get_writer(path, **kwargs):
        writer = Writer()
        writers.append(writer)
        return writer

    runtime = SimpleNamespace(
        RealtimeEpisode=SourceEpisode,
        OffScreenRenderEnv=lambda **kw: env,
        _target_entries=lambda item, env: [],
        _object_pose=lambda env, name, entry: {},
        imageio=SimpleNamespace(get_writer=get_writer),
    )
    legacy = SimpleNamespace(
        task_bddl=lambda suite, task: (
            SimpleNamespace(get_task_init_states=lambda task: [np.zeros(1)]),
            SimpleNamespace(language="pick cup"),
            "fake.bddl",
        ),
        init_diag_state=lambda env, args: {},
        update_preview_compatible_diagnostics=lambda *args: None,
        update_diagnostics=lambda *args: None,
        summarize_diagnostics=lambda diag, args: {"ok": True},
        resize_video_frame=lambda image, size: image,
    )
    source = SimpleNamespace(runtime=runtime, legacy=legacy)
    args = SimpleNamespace(
        output_dir=options.output,
        camera_size=4,
        control_freq=20,
        seed=0,
        init_state_id=0,
        video_fps=60.0,
    )
    item = {
        "item_id": "fake-item",
        "suite": "libero_spatial",
        "task_id": 0,
        "level": "l1",
        "traj_index": 0,
        "variant_id": "fake",
        "task_cfg": {},
    }
    return source, options, args, item, rpc, endpoints, env, writers


def make_runner(tmp_path, **kwargs):
    from coexecution.libero_dynamic.streaming_evaluation import (
        make_streaming_episode_type,
    )

    values = fixtures(tmp_path, **kwargs)
    source, options, args, item, rpc = values[:5]
    return make_streaming_episode_type(source, options)(item, args, rpc=rpc), values


def test_startup_moves_world_freshens_state_and_steady_actions_add_no_hold(tmp_path):
    runner, (_, options, _, _, _, endpoints, env, writers) = make_runner(tmp_path)
    result = runner.run()
    assert result["status"] == "ok"
    assert result["success"] is True
    assert result["steps"] == 3
    assert result["startup_seconds"] == 0.1
    assert result["rollout_sim_time_s"] == pytest.approx(0.25)
    assert env.sim.data.time == pytest.approx(2.25)
    assert env.sim.x == pytest.approx(0.06 * 2.25)
    assert env.fresh_calls.count((pytest.approx(2.1), True)) >= 1
    audit = json.loads(Path(result["policy_audit"]).read_text())
    assert len(audit["warmup"]) == 1
    assert audit["streaming"]["bootstrap"]["capture_s"] == 1000
    scored = [d for name, d in endpoints[1001].calls if name == "plan"][-3:]
    assert scored[0]["observation"]["state"]["time"] == pytest.approx(2.1)
    assert len(audit["video_samples"]) == 15
    assert {row["phase"] for row in audit["video_samples"]} == {
        "bootstrap_hold",
        "action",
    }
    assert all(len(w.frames) == 15 and w.closed for w in writers)
    assert writers[0].frames[0].shape == (4, 8, 3)
    assert writers[1].frames[0].shape == (4, 4, 3)
    assert all(a[6] == -1 for a in env.actions[40:])
    assert env.closed and all(e.closed for e in endpoints.values())
    assert options.output.exists()


def test_physical_failure_keeps_partial_actions_rpc_events_and_media(tmp_path):
    runner, (_, _, _, _, _, endpoints, env, writers) = make_runner(tmp_path)
    env.fail_after = 42
    result = runner.run()
    assert result["status"] == "error" and not result["success"]
    assert "physical step interrupted" in result["error"]
    assert result["steps"] == 1
    assert result["rollout_sim_time_s"] == pytest.approx(0.152)
    audit = json.loads(Path(result["policy_audit"]).read_text())
    assert len(audit["streaming"]["actions"]) == 1
    assert audit["action_attempts"][-1]["status"] == "error"
    assert audit["rpc_records"]
    assert audit["streaming"]["events"]
    assert env.closed and all(e.closed for e in endpoints.values())
    assert all(w.closed for w in writers)


def test_disabled_video_still_checks_actual_physics_and_motion(tmp_path):
    runner, (_, _, _, _, _, _, env, writers) = make_runner(tmp_path, videos=False)
    result = runner.run()
    assert result["status"] == "ok" and result["steps"] == 3
    assert writers == []
    audit = json.loads(Path(result["policy_audit"]).read_text())
    assert audit["video_samples"] == []
    assert (
        len([e for e in audit["motion_events"] if e["kind"] == "action_completed"]) == 3
    )
    assert env.closed


def test_native_fixture_callback_convention_does_not_double_offset_owned_steps(
    tmp_path, monkeypatch
):
    from coexecution.libero_dynamic import physical

    source, options, args, item, _, _, env, _ = fixtures(tmp_path)
    item["preview"] = False

    class Fixture:
        backend = "fixture"

        def __init__(self, env):
            self.env = env

        def advance(self, seconds, *, on_substep=None):
            self.env.sim.data.time += seconds
            if on_substep:
                on_substep(seconds)
            return seconds

    monkeypatch.setattr(physical, "NativeFixtureTiming", Fixture)
    for convention, expected_offset in (("source_run", 0.1), ("physical_origin", 0)):
        episode = physical.make_physical_episode_type(
            source, options, callback_convention=convention
        )(item, args)
        episode._setup_world(env, np.zeros(1))
        episode.control_started = True
        episode._apply_sync_latency(env, 0.1)
        samples = []
        episode._step_env(env, np.zeros(7), on_substep=samples.append)
        assert samples[0] == pytest.approx(0.002 + expected_offset)


def test_cli_adds_pir2_without_changing_existing_flow_arguments(
    tmp_path, monkeypatch, capsys
):
    from coexecution.libero_dynamic import __main__ as cli
    from coexecution.libero_dynamic import evaluation, streaming_evaluation

    observed = []
    monkeypatch.setattr(
        evaluation, "evaluate_default", lambda args: observed.append(args) or {}
    )
    monkeypatch.setattr(
        streaming_evaluation,
        "evaluate_streaming",
        lambda args: observed.append(args) or {},
    )
    common = [
        "--upstream",
        str(tmp_path),
        "--output",
        str(tmp_path / "out"),
        "--suite",
        "libero_spatial",
        "--speed-mps",
        ".06",
        "--expected-checkpoint",
        str(tmp_path),
        "--precision",
        "bf16",
    ]
    cli.main(["evaluate", *common, "--model-port", "1001", "--latency-seconds", ".1"])
    assert observed[-1].chunk_steps == 8
    cli.main(
        [
            "evaluate-pir2",
            *common,
            "--timing-mode",
            "fixed",
            "--action-port",
            "1001",
            "--vision-port",
            "1002",
            "--bootstrap-latency-seconds",
            ".1",
            "--s1-latency-seconds",
            ".05",
            "--s2-latency-seconds",
            ".1",
            "--recovery-latency-seconds",
            ".3",
        ]
    )
    assert observed[-1].warmup_requests == 8
    assert observed[-1].fixed_delay is None
    assert observed[-1].recovery_latency_seconds == 0.3


def test_warmup_covers_delays_and_never_advances_world(tmp_path):
    from coexecution.libero_dynamic.streaming_evaluation import warmup_streaming

    _, _, _, _, rpc, _, env, _ = fixtures(tmp_path)
    before = env.sim.data.time
    result = warmup_streaming(
        rpc,
        env._get_observations(),
        "pick cup",
        rounds=2,
        delays=[1, 2, 3, 4, 5],
        seed=0,
        capture_s=1000,
    )
    assert len(result) == 10
    assert [r["delay_ticks"] for r in result] == [1, 2, 3, 4, 5] * 2
    assert all(r["phase"] == "warmup" for r in result)
    assert env.sim.data.time == before


def test_warmup_failure_closes_both_policies_and_env_without_startup_time(tmp_path):
    runner, (_, _, _, _, _, endpoints, env, writers) = make_runner(tmp_path)
    endpoints[1001].mutate = lambda name, result: (
        result.update(forward_counts={"vlm": 0, "dit": 0})
        if name == "bootstrap"
        else None
    )
    result = runner.run()
    assert result["status"] == "error" and result["steps"] == 0
    assert result["rollout_sim_time_s"] == 0
    assert env.sim.data.time == pytest.approx(2)
    assert env.closed and all(endpoint.closed for endpoint in endpoints.values())
    assert writers == []
    audit = json.loads(Path(result["policy_audit"]).read_text())
    assert audit["warmup"][0]["status"] == "started"
    assert audit["rpc_records"][-1]["phase"] == "warmup"


def test_failed_camera_close_still_closes_other_camera_model_and_environment(tmp_path):
    runner, (source, _, _, _, _, endpoints, env, writers) = make_runner(tmp_path)
    old_open = source.runtime.imageio.get_writer

    def get_writer(*args, **kwargs):
        writer = old_open(*args, **kwargs)
        if len(writers) == 1:

            def failed_close():
                raise RuntimeError("failed camera flush")

            writer.close = failed_close
        return writer

    source.runtime.imageio.get_writer = get_writer
    result = runner.run()
    assert result["status"] == "error" and not result["success"]
    assert "failed camera flush" in result["error"]
    assert writers[1].closed
    assert env.closed and all(endpoint.closed for endpoint in endpoints.values())


def test_scored_rpc_phase_and_source_diagnostics_are_saved(tmp_path):
    runner, _ = make_runner(tmp_path)
    result = runner.run()
    audit = json.loads(Path(result["policy_audit"]).read_text())
    assert audit["diagnostics"] == {"ok": True}
    assert result["diagnostics"] == {"ok": True}
    assert audit["rpc_records"][-1]["phase"] == "scored"
    assert result["success_predicate"] == "source done or env.check_success"


@pytest.mark.parametrize("fail", [False, True, "interrupt"])
def test_batch_manifests_and_summary_preserve_success_or_failed_partial_episode(
    tmp_path, monkeypatch, fail
):
    from coexecution.libero_dynamic import streaming_evaluation as module

    source, options, args, item, rpc, _, env, _ = fixtures(tmp_path)
    options.output = tmp_path / "batch"
    options.upstream = tmp_path / "source"
    options.action_port, options.vision_port = 1001, 1002
    options.model_host = "127.0.0.1"
    args.output_dir = options.output
    if fail:
        env.fail_after = 42
    if fail == "interrupt":
        old_step = env.sim.step

        def interrupt_step():
            old_step()
            if len(env.actions) == 42:
                raise KeyboardInterrupt("cancel scored physics")

        env.sim.step = interrupt_step
    inventory = {
        "upstream": {"actual_revision": "test-owned-source"},
        "phase_contract": {},
    }
    monkeypatch.setattr(
        module,
        "prepare_conditions",
        lambda options: (inventory, [{"condition_id": "condition-1"}]),
    )
    monkeypatch.setattr(module, "_load_runtime", lambda path: source)
    monkeypatch.setattr(module, "_source_arguments", lambda source, options: args)
    monkeypatch.setattr(module, "_effective_items", lambda *values: [item])
    monkeypatch.setattr(module, "Pir2RPC", lambda **kwargs: rpc)
    if fail == "interrupt":
        with pytest.raises(KeyboardInterrupt, match="cancel scored physics"):
            module.evaluate_streaming(options)
    elif fail:
        with pytest.raises(RuntimeError, match="episode failed"):
            module.evaluate_streaming(options)
    else:
        assert module.evaluate_streaming(options)["success_rate"] == 1
    records = [
        json.loads(line)
        for line in (options.output / "manifest.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["condition_id"] == "condition-1"
    assert records[0]["success"] is (not fail)
    assert records[0]["upstream"] == inventory["upstream"]
    assert Path(records[0]["policy_audit"]).is_file()
    evaluation = json.loads((options.output / "evaluation.json").read_text())
    summary = json.loads((options.output / "evaluation-summary.json").read_text())
    assert (
        evaluation["status"] == summary["status"] == ("failed" if fail else "complete")
    )
    assert summary["conditions_completed"] == 1


def test_native_setup_failure_stops_before_any_scored_rpc(tmp_path):
    runner, (_, _, _, _, _, endpoints, env, _) = make_runner(tmp_path)
    # Interrupt native setup; its physical error must not become a model result.
    env.fail_after = 20
    before = len(endpoints[1001].calls)
    result = runner.run()
    assert result["status"] == "error" and result["steps"] == 0
    assert len(endpoints[1001].calls) == before
    assert env.closed


def test_startup_actual_clock_mismatch_stops_before_steady_plan(tmp_path):
    runner, (_, _, _, _, _, endpoints, env, _) = make_runner(tmp_path)
    original = runner._apply_sync_latency

    def wrong_hold(env, duration):
        result = original(env, duration)
        env.sim.step()
        return result

    runner._apply_sync_latency = wrong_hold
    result = runner.run()
    assert result["status"] == "error" and result["steps"] == 0
    assert "startup" in result["error"]
    plans = [data for name, data in endpoints[1001].calls if name == "plan"]
    assert len(plans) == 1  # warmup only
    assert env.closed


def test_cli_help_declares_native_default_independent_lanes_and_serial_work(capsys):
    from coexecution.libero_dynamic.__main__ import main

    with pytest.raises(SystemExit) as caught:
        main(["evaluate-pir2", "--help"])
    assert caught.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "native per-request latency by default" in help_text
    assert "fixed counterfactual timing is an explicit control" in help_text
    assert "Real GPU/physics work is serial" in help_text
    assert "independent logical service lanes" in help_text
    assert "--timing-mode {native,fixed}" in help_text
    for stage in ("bootstrap", "s1", "s2", "recovery"):
        assert f"Fixed counterfactual {stage} service duration in seconds" in help_text


@pytest.mark.parametrize("failure", ["success", "diagnostics"])
def test_completed_physical_action_is_counted_when_poststep_processing_fails(
    tmp_path, failure
):
    runner, (source, _, _, _, _, _, env, _) = make_runner(tmp_path)

    def fail(*args):
        raise RuntimeError("post-step failure")

    if failure == "success":
        env.check_success = fail
    else:
        source.legacy.update_preview_compatible_diagnostics = fail
    result = runner.run()
    assert result["status"] == "error"
    assert result["steps"] == 1
    assert result["controller_recorded_steps"] == 0
    audit = json.loads(Path(result["policy_audit"]).read_text())
    assert audit["action_attempts"][0]["physics_complete"] is True
    assert audit["action_attempts"][0]["physics_elapsed_s"] == pytest.approx(0.05)
    assert audit["streaming"]["actions"] == []


def test_dual_written_frame_keeps_mapping_when_wrist_append_fails(tmp_path):
    runner, (source, _, _, _, _, _, _, writers) = make_runner(tmp_path)
    original_open = source.runtime.imageio.get_writer

    def open_writer(*args, **kwargs):
        writer = original_open(*args, **kwargs)
        if len(writers) == 2:
            original_append = writer.append_data

            def append(frame):
                if len(writer.frames) == 3:
                    raise RuntimeError("wrist encoder failure")
                original_append(frame)

            writer.append_data = append
        return writer

    source.runtime.imageio.get_writer = open_writer
    result = runner.run()
    assert result["status"] == "error" and result["steps"] == 0
    audit = json.loads(Path(result["policy_audit"]).read_text())
    assert audit["video_frame_counts"] == [4, 3]
    assert len(audit["video_samples"]) == 4
    sample = audit["video_samples"][3]
    assert sample["frame_index"] == 3
    assert sample["sample_sim_time_s"] == pytest.approx(0.068)
    assert sample["streams"]["dual"]["status"] == "written"
    assert sample["streams"]["dual"]["frame_index"] == 3
    assert sample["streams"]["wrist"]["status"] == "error"
    assert result["video_frame_counts"] == {"dual": 4, "wrist": 3}
    assert all(writer.closed for writer in writers)


def test_cli_native_default_needs_no_fixed_latency_arguments(tmp_path):
    import argparse

    from coexecution.libero_dynamic.streaming_evaluation import (
        add_streaming_arguments,
        stage_latencies,
    )

    parser = argparse.ArgumentParser()
    add_streaming_arguments(parser)
    args = parser.parse_args(
        [
            "--upstream",
            str(tmp_path),
            "--output",
            str(tmp_path / "new"),
            "--expected-checkpoint",
            str(tmp_path),
            "--action-port",
            "1001",
            "--vision-port",
            "1002",
            "--precision",
            "bf16",
            "--suite",
            "libero_spatial",
            "--speed-mps",
            ".06",
        ]
    )
    assert args.timing_mode == "native"
    assert stage_latencies(args) is None


@pytest.mark.parametrize(
    "mode, values, valid",
    [
        ("native", (None, None, None, None), True),
        ("native", (0.1, None, None, None), False),
        ("fixed", (0.1, 0.05, 0.1, 0.2), True),
        ("fixed", (0.1, 0.05, 0.1, None), False),
        ("unknown", (None, None, None, None), False),
    ],
)
def test_timing_mode_rejects_conflicting_or_missing_latency_options(
    mode, values, valid
):
    from coexecution.libero_dynamic.streaming_evaluation import stage_latencies

    options = SimpleNamespace(
        timing_mode=mode,
        **{
            stage + "_latency_seconds": value
            for stage, value in zip(
                ("bootstrap", "s1", "s2", "recovery"), values, strict=True
            )
        },
    )
    if valid:
        result = stage_latencies(options)
        assert (result is None) == (mode == "native")
    else:
        with pytest.raises(ValueError):
            stage_latencies(options)


def test_native_startup_pays_measured_bootstrap_once_in_physical_world(
    tmp_path, monkeypatch
):
    from coexecution.libero_dynamic import streaming_evaluation as module
    from tests.contracts.test_dynamic_streaming import native_setup

    source, options, args, item, _, _, env, _ = fixtures(tmp_path)
    options.timing_mode = "native"
    options.warmup_requests = 0
    for name in ("bootstrap", "s1", "s2", "recovery"):
        setattr(options, name + "_latency_seconds", None)
    _, rpc, _, _, clock, _ = native_setup(tmp_path, bootstrap=0.137)
    original = module.StreamingController
    monkeypatch.setattr(
        module, "StreamingController", lambda *a, **kw: original(*a, clock=clock, **kw)
    )
    result = module.make_streaming_episode_type(source, options)(
        item, args, rpc=rpc
    ).run()
    assert result["status"] == "ok"
    assert result["steps"] == 3
    assert result["startup_seconds"] == pytest.approx(0.137)
    assert result["startup_seconds_applied"] == pytest.approx(0.137)
    assert result["rollout_sim_time_s"] == pytest.approx(0.287)
    assert env.sim.x == pytest.approx(0.06 * 2.287)
    assert result["timing_mode"] == "native"
    assert result["stage_latencies"] is None
    audit = json.loads(Path(result["policy_audit"]).read_text())
    assert audit["bootstrap"]["logical_duration_s"] == result["startup_seconds"]
    assert audit["streaming"]["startup_seconds"] == result["startup_seconds"]
    assert result["control_origin_s"] == pytest.approx(1000.137)


@pytest.mark.parametrize(
    "mode, latency_args",
    [
        ("native", ["--s1-latency-seconds", ".04"]),
        ("fixed", ["--s1-latency-seconds", ".04"]),
    ],
)
def test_real_cli_rejects_conflicting_timing_before_loading_runtime(
    tmp_path, monkeypatch, mode, latency_args
):
    from coexecution.libero_dynamic import streaming_evaluation
    from coexecution.libero_dynamic.__main__ import main

    def must_not_load(*args):
        pytest.fail("invalid timing must be rejected before accessing upstream")

    monkeypatch.setattr(streaming_evaluation, "prepare_conditions", must_not_load)
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "evaluate-pir2",
                "--upstream",
                str(tmp_path),
                "--output",
                str(tmp_path / "new"),
                "--suite",
                "libero_spatial",
                "--speed-mps",
                ".06",
                "--expected-checkpoint",
                str(tmp_path),
                "--action-port",
                "1001",
                "--vision-port",
                "1002",
                "--precision",
                "bf16",
                "--timing-mode",
                mode,
                *latency_args,
            ]
        )
    assert caught.value.code == 2
    assert not (tmp_path / "new").exists()


def test_native_failure_before_receipt_does_not_invent_paid_startup(
    tmp_path, monkeypatch
):
    from coexecution.libero_dynamic import streaming_evaluation as module
    from tests.contracts.test_dynamic_streaming import native_setup

    source, options, args, item, _, _, env, _ = fixtures(tmp_path)
    options.timing_mode = "native"
    options.warmup_requests = 0
    for name in ("bootstrap", "s1", "s2", "recovery"):
        setattr(options, name + "_latency_seconds", None)
    _, rpc, endpoints, _, clock, _ = native_setup(tmp_path)

    def fail_bootstrap(endpoint, result):
        if endpoint == "bootstrap":
            raise RuntimeError("failed scored bootstrap")

    endpoints[1001].mutate = fail_bootstrap
    original = module.StreamingController
    monkeypatch.setattr(
        module, "StreamingController", lambda *a, **kw: original(*a, clock=clock, **kw)
    )
    result = module.make_streaming_episode_type(source, options)(
        item, args, rpc=rpc
    ).run()
    assert result["status"] == "error"
    assert result["startup_seconds"] is None
    assert result["control_origin_s"] is None
    assert result["startup_seconds_applied"] == result["rollout_sim_time_s"] == 0
    assert env.sim.data.time == pytest.approx(2.0)
    audit = json.loads(Path(result["policy_audit"]).read_text())
    assert audit["bootstrap"] is None
    assert audit["streaming"]["timing_mode"] == "native"
    assert audit["streaming"]["stage_latencies"] is None
    assert audit["rpc_records"][-1]["status"] == "error"
    assert audit["cleanup_errors"] == []


@pytest.mark.parametrize("timing_mode", ["native", "fixed"])
def test_streaming_cli_remains_compatible_with_shared_source_arguments(
    tmp_path, timing_mode
):
    import argparse
    import copy

    from coexecution.libero_dynamic.evaluation import _source_arguments
    from coexecution.libero_dynamic.streaming_evaluation import (
        add_streaming_arguments,
        validate_options,
    )

    parser = argparse.ArgumentParser()
    add_streaming_arguments(parser)
    argv = [
        "--upstream",
        str(tmp_path),
        "--output",
        str(tmp_path / "out"),
        "--expected-checkpoint",
        str(tmp_path),
        "--suite",
        "libero_spatial",
        "--precision",
        "bf16",
        "--action-port",
        "1001",
        "--vision-port",
        "1002",
        "--speed-mps",
        "0.06",
        "--timing-mode",
        timing_mode,
    ]
    if timing_mode == "fixed":
        for stage, seconds in [
            ("bootstrap", "0.2"),
            ("s1", "0.04"),
            ("s2", "0.15"),
            ("recovery", "0.2"),
        ]:
            argv.extend([f"--{stage}-latency-seconds", seconds])
    options = parser.parse_args(argv)
    validate_options(options)
    assert not hasattr(options, "native_latency")
    source_options = copy.copy(options)
    source_options.chunk_steps = 1  # Same adapter used by evaluate_streaming.
    source_parser = argparse.ArgumentParser()
    source_parser.add_argument("--output-dir")
    source = SimpleNamespace(
        runner=SimpleNamespace(build_parser=lambda: source_parser),
        legacy=SimpleNamespace(apply_reviewed_old_physics_profile=lambda args: None),
    )
    runtime_args = _source_arguments(source, source_options)
    assert runtime_args.replan_steps == 1
    # StreamingController owns per-stage timing; the shared upstream setup has
    # no Flow request sampler to activate, regardless of native/fixed piR2 mode.
    assert runtime_args.latency_source == "fixed"
    assert runtime_args.calibrated_latency_seconds == 0
    assert options.timing_mode == timing_mode
    assert not hasattr(options, "native_latency")
