"""Host-only responsive budget and wakeable controller contracts."""

import importlib
import math

import pytest

from coexecution.libero import scheduling


def policy(initial=3):
    assert hasattr(scheduling, "ResponsiveDelayBudget")
    return scheduling.ResponsiveDelayBudget(initial, 0.05)


def observe(budget, duration=0.025, dispatch=0, request_delay=None):
    d = budget.delay if request_delay is None else request_delay
    return budget.observe(
        submitted_s=10 + dispatch,
        completed_s=10 + dispatch + duration,
        scheduled_s=10,
        deadline_s=10 + d * 0.05,
    )


def test_full_window_and_three_observations_before_one_tick_recovery():
    budget = policy()
    for _ in range(9):
        assert observe(budget)["chosen_delay_ticks"] == 3
    decision = observe(budget)
    assert decision["chosen_delay_ticks"] == 2
    assert decision["reason"] == "lower_full_window_hysteresis"
    assert len(decision["samples_seconds"]) == 8
    for _ in range(2):
        assert observe(budget)["chosen_delay_ticks"] == 2
    assert observe(budget)["chosen_delay_ticks"] == 1


def test_latency_rise_and_dispatch_lateness_raise_immediately():
    budget = policy(1)
    assert observe(budget, 0.113)["chosen_delay_ticks"] == 3
    late = observe(budget, 0.04, dispatch=0.12)
    assert late["chosen_delay_ticks"] == 4
    assert late["deadline_miss"]
    assert late["scheduled_required_ticks"] == 4


def test_slow_window_prevents_premature_recovery_then_fast_window_recovers():
    budget = policy(3)
    for _ in range(12):
        assert observe(budget, 0.12)["chosen_delay_ticks"] == 3
    for _ in range(8):
        assert observe(budget)["chosen_delay_ticks"] == 3
    assert observe(budget)["chosen_delay_ticks"] == 2


def test_bounds_invalid_samples_and_episode_reset():
    budget = policy(1)
    high = observe(budget, 0.5)
    assert high["chosen_delay_ticks"] == 5
    assert high["over_training_budget"]
    for value in (math.nan, math.inf, -0.1):
        assert not observe(budget, value)["sample_valid"]
    assert len(budget.samples) == 1
    assert policy(2).samples == ()
    for value in (0, 6, True):
        with pytest.raises(ValueError):
            policy(value)


def test_default_and_responsive_configuration_constraints():
    evaluator = importlib.import_module("scripts.evaluate_libero_protocol")
    parser = evaluator.build_parser()
    base = [
        "--protocol",
        "deployment",
        "--variant",
        "pir2",
        "--output",
        "/tmp/responsive-test",
    ]
    args = parser.parse_args(base)
    assert getattr(args, "deployment_scheduling_policy", None) == "conservative"
    for flags in (
        ["--variant", "flow"],
        ["--protocol", "algorithm", "--variant", "pir2"],
        ["--variant", "pir2", "--deployment-fixed-delay", "1"],
        ["--variant", "pir2", "--execution-clock", "ready"],
        ["--variant", "pir2", "--execution-coupling", "synchronous"],
    ):
        args = parser.parse_args(
            [*base, "--deployment-scheduling-policy", "responsive", *flags]
        )
        with pytest.raises(ValueError):
            evaluator.validate_deployment_controls(args)
    args = parser.parse_args([*base, "--deployment-scheduling-policy", "responsive"])
    evaluator.validate_deployment_controls(args)


class ReadyClock:
    def __init__(self):
        self.t = 10.0
        self.events = []
        self.future = None

    def now(self):
        return self.t

    def sleep_until(self, target):
        assert target >= 0
        self.t = max(self.t, target)

    def wait_for_signal(self, signal, deadline):
        assert deadline >= self.t
        self.sleep_until(min(deadline, self.future.request["complete"]))
        if self.future.done():
            self.future.callback(self.future)


def control_case(
    *,
    policy_name="responsive",
    latency=0.02,
    step_seconds=0.01,
    terminal=False,
    steps=5,
    initial_delay=1,
    error=False,
):
    import io
    import json

    import numpy as np

    from scripts.libero_wallclock import run_control_loop
    from scripts.test_libero_wallclock import Actor, Future, Vision

    clock = ReadyClock()

    class ReadyFuture(Future):
        def add_done_callback(self, fn):
            self.callback = fn
            if self.done():
                fn(self)

        def result(self):
            if error:
                raise RuntimeError("inference failed")
            return super().result()

    class ReadyActor(Actor):
        closed = False

        def __init__(self):
            super().__init__(clock, latency)
            self.submissions = []

        def submit(self, obs, r, d, prefix, visual, deadline, submitted):
            self.submissions.append((r, d, submitted, obs.tick, obs.capture_s))
            clock.future = ReadyFuture(
                self,
                {
                    "tick": r,
                    "d": d,
                    "start": submitted,
                    "complete": submitted + self.latency,
                    "deadline": deadline,
                },
            )
            return clock.future

        def close(self):
            self.closed = True

    actor = ReadyActor()
    vision = Vision(clock)
    physics = []

    def step(action):
        physics.append(clock.now())
        clock.t += step_seconds
        return {"state.x": [len(physics)]}, terminal, False

    trace = io.StringIO()
    kwargs = {
        "step_fn": step,
        "initial": {"state.x": [0]},
        "initial_capture_s": 10.0,
        "initial_plan": np.zeros((40, 7)),
        "actor": actor,
        "vision": vision,
        "initial_delay": initial_delay,
        "period": 0.05,
        "max_steps": steps,
        "trace": trace,
        "clock": clock,
        "deployment_scheduling_policy": policy_name,
    }
    if error:
        with pytest.raises(RuntimeError, match="inference failed"):
            run_control_loop(**kwargs)
        assert actor.closed
        assert any(event[0] == "stop_vision" for event in clock.events)
        return
    result, _ = run_control_loop(**kwargs)
    return (
        result,
        [json.loads(line) for line in trace.getvalue().splitlines()],
        physics,
        actor,
        vision,
    )


@pytest.mark.parametrize("step_seconds, expected", [(0.01, 10.02), (0.03, 10.03)])
def test_publish_at_first_safe_boundary_without_early_physics_or_sensors(
    step_seconds, expected
):
    _, rows, physics, actor, vision = control_case(step_seconds=step_seconds)
    publication = next(r for r in rows if r["kind"] == "request_completed")
    assert publication["t_s"] == pytest.approx(expected)
    assert publication["consumed_at_tick"] == 1
    assert publication["installed_slots"] == 1
    assert publication["controller_ready_observed_s"] == pytest.approx(expected)
    assert publication["action_buffer"]["slots"][0]["tick"] == 1
    assert physics == pytest.approx([10 + i * 0.05 for i in range(5)])
    assert all(at >= 10 + tick * 0.05 for tick, at in vision.offers)
    assert all(
        submitted >= 10 + source * 0.05
        for _, _, submitted, source, _ in actor.submissions
    )


def test_recovery_never_advances_previous_backend_anchor():
    result, _, _, actor, _ = control_case(initial_delay=3, steps=50)
    assert result["final_action_delay_ticks"] == 1
    for previous, following in zip(actor.submissions, actor.submissions[1:]):
        assert following[0] >= previous[0] + previous[1]


def test_terminal_result_is_unused_and_default_keeps_tick_boundary_ingestion():
    _, rows, _, _, _ = control_case(terminal=True)
    assert not any(r["kind"] == "request_completed" for r in rows)
    assert any(r["kind"] == "unused_result_at_episode_end" for r in rows)
    _, rows, _, _, _ = control_case(policy_name="conservative")
    assert next(r for r in rows if r["kind"] == "request_completed")[
        "t_s"
    ] == pytest.approx(10.05)


def test_ready_error_releases_workers():
    control_case(error=True)


def test_incomplete_request_never_blocks_past_physical_deadline():
    _, rows, physics, _, _ = control_case(latency=0.113, steps=8)
    assert physics == pytest.approx([10 + i * 0.05 for i in range(8)])
    assert next(r for r in rows if r["kind"] == "request_completed")[
        "t_s"
    ] == pytest.approx(10.113)
    assert (
        next(r for r in rows if r["kind"] == "delay_policy_decision")[
            "chosen_delay_ticks"
        ]
        == 3
    )


def test_future_callback_signals_without_reading_or_mutating_controller():
    from concurrent.futures import Future

    future = Future()
    signal = scheduling.FutureSignal(future)
    assert not signal.signal.is_set()
    future.set_result({"actions": "opaque"})
    assert signal.signal.is_set()
    assert signal.wait_until(ReadyClock(), 10)


def test_quantile_and_rounding_do_not_lower_below_one_tick():
    budget = policy(1)
    for _ in range(12):
        assert observe(budget, 0.045)["chosen_delay_ticks"] == 1
    assert observe(budget, 0.05)["candidate_delay_ticks"] == 2
