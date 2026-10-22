"""Logical asynchronous dynamic control; doubles perform no neural inference."""

from copy import deepcopy

import numpy as np
import pytest


def api():
    from coexecution.libero_dynamic.streaming import (
        StageLatencies,
        StepResult,
        StreamingController,
    )
    from coexecution.libero_dynamic.streaming_rpc import Pir2RPC

    return StageLatencies, StepResult, StreamingController, Pir2RPC


class Endpoint:
    def __init__(self, role, checkpoint):
        self.role = role
        self.identity = {
            "variant": "pir2",
            "role": role,
            "checkpoint": str(checkpoint),
            "sha256": {"model.safetensors": "a" * 64},
            "predicted_horizon": 40,
            "decoded_horizon": 40,
            "feature_transport": "numpy_bf16",
            "feature_rpc_modes": ["combined"],
            "policy_instance": role,
            "physical_device_uuid": "gpu",
            "execution_endpoints": ["bootstrap", "plan"],
        }
        self.calls = []
        self.closed = False
        self.anchor = 0
        self.cache = None
        self.mutate = None

    def call_endpoint(self, endpoint, data=None, requires_input=True):
        self.calls.append((endpoint, deepcopy(data)))
        if endpoint == "identity":
            return deepcopy(self.identity)
        if endpoint == "reset":
            return {"reset": True}
        if endpoint == "bootstrap":
            self.anchor = 0
            self.cache = {"capture_s": data["capture_s"], "source_tick": 0}
            actions = np.zeros((40, 7), np.float32)
            actions[:, 0] = 0.1
            actions[:, 6] = 1
            result = {"actions": actions, "forward_counts": {"vlm": 1, "dit": 4}}
        elif endpoint == "vision":
            result = {
                "vl_embeds": {
                    "schema": "libero.features",
                    "version": 1,
                    "encoding": "numpy_bf16",
                    "fields": {
                        "private_feature": {
                            "dtype": "int64",
                            "shape": [1],
                            "data": np.array([data["source_tick"]], dtype=np.int64),
                        }
                    },
                },
                "meta": {
                    "capture_s": data["capture_s"],
                    "source_tick": data["source_tick"],
                    "vlm_forward_calls": 1,
                    "frame_hashes": {"image": str(data["observation"]["frame"])},
                },
            }
        else:
            assert endpoint == "plan"
            r, d = data["request_tick"], data["delay_ticks"]
            if "feature_meta" in data:
                self.cache = deepcopy(data["feature_meta"])
            recovered = r - self.anchor >= 40
            shift = r - self.anchor
            self.anchor = r + d
            actions = np.zeros((d, 7), np.float32)
            actions[:, 0] = 0.9
            actions[:, 6] = 1
            result = {
                "actions": actions,
                "audit": {
                    "request_tick": r,
                    "d_request": d,
                    "valid_start_tick": r + d,
                    "valid_end_tick": r + 2 * d,
                    "buffer_origin_after": r + d,
                    "extra_buffer_shift": shift,
                    "buffer_restarted": recovered,
                    "state_capture_s": data["state_capture_s"],
                    "cache": deepcopy(self.cache),
                    "image_delay_ticks": round(
                        (data["state_capture_s"] - self.cache["capture_s"]) / 0.05
                    ),
                    "forward_counts": {"vlm": 0, "dit": 5 if recovered else 1},
                    "server_plan_started_s": 900000.0,
                },
            }
        if self.mutate:
            self.mutate(endpoint, result)
        return result

    def close(self):
        self.closed = True


def setup(tmp_path, *, s1=0.05, s2=0.1, bootstrap=0.1, recovery=0.3, fixed=1):
    Latencies, Step, Controller, RPC = api()
    checkpoint = tmp_path / "existing"
    checkpoint.mkdir(exist_ok=True)
    endpoints = {
        1001: Endpoint("action", checkpoint),
        1002: Endpoint("vlm", checkpoint),
    }
    rpc = RPC(
        action_port=1001,
        vision_port=1002,
        expected_checkpoint=checkpoint,
        precision="bf16",
        client_factory=lambda **kw: endpoints[kw["port"]],
        preprocessor=lambda obs, language: deepcopy(obs),
    )
    controller = Controller(
        rpc,
        latencies=Latencies(
            bootstrap_s=bootstrap, s1_s=s1, s2_s=s2, recovery_s=recovery
        ),
        fixed_delay=fixed,
    )
    receipt = controller.bootstrap(
        {"frame": -1, "state": {"tick": -1}}, "pick cup", seed=1
    )
    steps = []

    def step(action, tick):
        steps.append((tick, action.copy()))
        return Step({"frame": tick + 1, "state": {"tick": tick + 1}})

    def run(n, step_fn=step):
        return controller.run(
            receipt,
            initial_observation={"frame": 0, "state": {"tick": 0}},
            initial_capture_s=receipt["capture_s"] + bootstrap,
            language="pick cup",
            step_fn=step_fn,
            max_steps=n,
        )

    return controller, rpc, endpoints, receipt, steps, run


def requests(result, lane):
    return [record for record in result["requests"] if record["lane"] == lane]


def test_equality_completion_publishes_before_boundary_execution(tmp_path):
    _, _, _, receipt, steps, run = setup(tmp_path)
    result = run(3)
    assert len(steps) == 3
    assert result["actions"][0]["source"]["producer_kind"] == "bootstrap"
    assert result["actions"][1]["source"]["producer_request_tick"] == 0
    assert steps[1][1][0] == pytest.approx(0.9)
    assert steps[1][1][6] == -1
    assert receipt["capture_s"] > 0
    assert result["startup_seconds"] == 0.1
    assert result["steady_seconds"] == pytest.approx(0.15)
    assert result["physical_seconds"] == pytest.approx(0.25)


def test_instant_gpu_result_is_private_until_logical_s2_completion(tmp_path):
    _, _, endpoints, _, _, run = setup(tmp_path, s2=0.12)
    result = run(4)
    plans = [d for name, d in endpoints[1001].calls if name == "plan"]
    assert [d.get("feature_meta", {}).get("source_tick") for d in plans] == [
        None,
        None,
        None,
        0,
    ]
    assert requests(result, "s2")[0]["logical_completed_s"] == pytest.approx(1000.22)
    assert requests(result, "s1")[0]["feature_age_s"] == pytest.approx(0.1)


def test_poststep_future_observation_is_not_visible_to_midstep_restart(tmp_path):
    _, _, endpoints, _, _, run = setup(tmp_path, s2=0.12)
    result = run(6)
    visions = [d for name, d in endpoints[1002].calls if name == "vision"]
    # At 120 ms the pending tick 2 image exists; tick 3 was host-computed but is private.
    assert [d["source_tick"] for d in visions][:3] == [0, 2, 4]
    assert [d["observation"]["frame"] for d in visions][:3] == [0, 2, 4]
    assert any(e["kind"] == "camera_replaced" for e in result["events"])


def test_zero_s2_finishes_once_per_offered_frame(tmp_path):
    _, _, _, _, _, run = setup(tmp_path, s2=0)
    result = run(4)
    assert len(requests(result, "s2")) == 4
    assert [x["feature_source_tick"] for x in requests(result, "s1")] == [0, 1, 2, 3]


def test_startup_image_is_old_but_first_plan_state_is_fresh(tmp_path):
    _, _, endpoints, _, _, run = setup(tmp_path, bootstrap=0.17, s2=0.2)
    result = run(1)
    bootstrap = next(d for name, d in endpoints[1001].calls if name == "bootstrap")
    plan = next(d for name, d in endpoints[1001].calls if name == "plan")
    assert bootstrap["observation"]["state"]["tick"] == -1
    assert plan["observation"]["state"]["tick"] == 0
    assert plan["state_capture_s"] == pytest.approx(1000.17)
    assert requests(result, "s1")[0]["feature_age_s"] == pytest.approx(0.17)
    assert result["actions"][0]["feature_age_s"] == pytest.approx(0.17)
    assert requests(result, "s1")[0]["server_audit"]["server_plan_started_s"] == 900000
    assert requests(result, "s1")[0]["feature_age_s"] < 1


def test_late_actions_expire_in_place_and_recovery_uses_its_own_latency(tmp_path):
    _, _, _, _, steps, run = setup(tmp_path, s1=2.2, s2=0.1, recovery=0.35)
    result = run(55)
    plans = requests(result, "s1")
    assert plans[0]["publication"]["expired_slots"] == 1
    assert plans[0]["publication"]["installed_slots"] == 0
    assert plans[1]["server_audit"]["buffer_restarted"] is True
    assert plans[1]["logical_duration_s"] == 0.35
    assert plans[1]["server_audit"]["forward_counts"] == {"vlm": 0, "dit": 5}
    assert result["actions"][40]["fallback"] is True
    assert steps[40][1][:6].tolist() == [0] * 6
    assert steps[40][1][6] == -1


def test_committed_prefix_and_output_interval_are_not_retimed(tmp_path):
    _, _, endpoints, _, _, run = setup(tmp_path, s1=0.075, fixed=2)
    result = run(6)
    plans = [d for name, d in endpoints[1001].calls if name == "plan"]
    np.testing.assert_array_equal(
        plans[0]["committed_actions"][:, 0], np.array([0.1, 0.1], dtype=np.float32)
    )
    assert [x["source"]["producer_kind"] for x in result["actions"][:2]] == [
        "bootstrap"
    ] * 2
    assert result["actions"][2]["source"]["producer_request_tick"] == 0
    assert requests(result, "s1")[0]["server_audit"]["valid_start_tick"] == 2


def test_success_ends_without_advancing_inflight_or_executing_future_slots(tmp_path):
    _, Step, _, _ = api()
    controller, rpc, endpoints, _, _, run = setup(tmp_path, s1=0.3, s2=0.4)
    result = run(10, lambda action, tick: Step({"frame": 1, "state": {}}, success=True))
    assert result["success"] is True
    assert result["action_steps"] == 1
    assert result["steady_seconds"] == 0.05
    assert {x["lane"] for x in result["inflight_at_end"]} == {"s1", "s2"}
    assert all(
        x["status"] == "unapplied_at_control_end" for x in result["inflight_at_end"]
    )
    before = controller.logical_time_s
    rpc.close()
    assert controller.logical_time_s == before
    assert all(e.closed for e in endpoints.values())


@pytest.mark.parametrize(
    "mutation",
    [
        {"variant": "flow"},
        {"decoded_horizon": 16},
        {"predicted_horizon": 16},
        {"feature_transport": "numpy_fp32"},
        {"feature_rpc_modes": ["separate"]},
        {"checkpoint": "/wrong/checkpoint"},
        {"sha256": {}},
        {"inference_optimization": {"config": {"precision": "w8a8"}}},
    ],
)
def test_prepare_rejects_incompatible_real_endpoint_identity(tmp_path, mutation):
    _, _, _, RPC = api()
    checkpoint = tmp_path
    endpoints = {1: Endpoint("action", checkpoint), 2: Endpoint("vlm", checkpoint)}
    endpoints[1].identity.update(mutation)
    rpc = RPC(
        action_port=1,
        vision_port=2,
        expected_checkpoint=checkpoint,
        precision="bf16",
        client_factory=lambda **kw: endpoints[kw["port"]],
    )
    with pytest.raises(ValueError):
        rpc.prepare()
    rpc.close()


def test_prepare_rejects_same_policy_instance_and_checkpoint_mismatch(tmp_path):
    _, _, _, RPC = api()
    endpoints = {1: Endpoint("action", tmp_path), 2: Endpoint("vlm", tmp_path)}
    endpoints[2].identity["policy_instance"] = "action"
    rpc = RPC(
        action_port=1,
        vision_port=2,
        expected_checkpoint=tmp_path,
        precision="bf16",
        client_factory=lambda **kw: endpoints[kw["port"]],
    )
    with pytest.raises(ValueError, match="independent"):
        rpc.prepare()
    rpc.close()


@pytest.mark.parametrize("endpoint", ["bootstrap", "vision", "plan"])
def test_real_rpc_forward_audits_are_required(tmp_path, endpoint):
    controller, _, endpoints, _, _, run = setup(tmp_path, s2=0)
    owner = endpoints[1002 if endpoint == "vision" else 1001]

    def mutate(name, result):
        if name == endpoint:
            if endpoint == "vision":
                result["meta"]["vlm_forward_calls"] = 0
            else:
                result.get("audit", result)["forward_counts"] = {"vlm": 0, "dit": 0}

    owner.mutate = mutate
    with pytest.raises(ValueError, match="forward"):
        if endpoint == "bootstrap":
            controller.bootstrap({"frame": 0, "state": {}}, "pick cup", seed=1)
        else:
            run(1)


def test_poststartup_timestamp_must_match_paid_hold(tmp_path):
    controller, _, _, receipt, _, _ = setup(tmp_path)
    _, Step, _, _ = api()
    with pytest.raises(ValueError, match="startup"):
        controller.run(
            receipt,
            initial_observation={"state": {}},
            initial_capture_s=receipt["capture_s"],
            language="pick cup",
            step_fn=lambda a, t: Step({}),
            max_steps=1,
        )


def test_responsive_budget_uses_logical_service_time_not_serial_rpc_wall(tmp_path):
    _, _, _, _, _, run = setup(tmp_path, s1=0.051, fixed=None)
    result = run(5)
    plans = requests(result, "s1")
    assert plans[0]["delay_ticks"] == 2
    assert plans[0]["budget_decision"]["sample_seconds"] == pytest.approx(0.051)
    assert plans[0]["budget_decision"]["sample_seconds"] != plans[0]["wall_duration_s"]


@pytest.mark.parametrize(
    "seconds,producer",
    [(0.01, 0), (0.049, 0), (0.05, 0), (0.051, None), (0.12, None), (0.26, None)],
)
def test_first_target_tick_adopts_only_results_ready_at_boundary(
    tmp_path, seconds, producer
):
    _, _, _, _, _, run = setup(tmp_path, s1=seconds)
    result = run(4)
    source = result["actions"][1]["source"]
    assert source["producer_request_tick"] == producer
    if seconds > 0.05:
        assert source["producer_kind"] == "bootstrap"
        assert (
            requests(result, "s1")[0]["publication"]["expired_slots"] == 1
            if seconds <= 0.2
            else True
        )


def test_wire_mismatch_is_rejected_before_feature_publication(tmp_path):
    _, _, endpoints, _, _, run = setup(tmp_path, s2=0)
    endpoints[1002].mutate = lambda name, result: (
        result.update(vl_embeds={"x": np.ones(3)}) if name == "vision" else None
    )
    with pytest.raises(ValueError, match="numpy_bf16"):
        run(1)


def test_recovery_claim_and_delay_age_must_match_actual_request_history(tmp_path):
    _, _, endpoints, _, _, run = setup(tmp_path)

    def mutate(name, result):
        if name == "plan":
            result["audit"].update(
                buffer_restarted=True, forward_counts={"vlm": 0, "dit": 5}
            )

    endpoints[1001].mutate = mutate
    with pytest.raises(ValueError, match="rolling"):
        run(1)


def test_image_age_from_server_must_use_the_same_logical_capture_domain(tmp_path):
    _, _, endpoints, _, _, run = setup(tmp_path)
    endpoints[1001].mutate = lambda name, result: (
        result["audit"].update(image_delay_ticks=99999) if name == "plan" else None
    )
    with pytest.raises(ValueError, match="image age"):
        run(1)


def test_raw_ages_above_training_delay_remain_visible(tmp_path):
    _, _, _, _, _, run = setup(tmp_path, s2=1, bootstrap=0.4)
    result = run(4)
    request = requests(result, "s1")[0]
    assert request["feature_age_s"] == pytest.approx(0.4)
    assert request["image_delay_ticks"] == 8
    assert request["image_delay_embedding_ticks"] == 5


def test_actions_and_prefix_copies_do_not_allow_step_to_mutate_queue(tmp_path):
    _, Step, _, _ = api()
    _, _, endpoints, _, _, run = setup(tmp_path)

    def step(action, tick):
        action[:] = 999
        return Step({"frame": tick + 1, "state": {}})

    result = run(3, step)
    assert all(x["raw_action"][6] == -1 for x in result["actions"])
    plans = [d for name, d in endpoints[1001].calls if name == "plan"]
    assert all(np.max(x["committed_actions"]) <= 1 for x in plans)


def test_close_and_model_calls_retain_thread_affinity(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    _, rpc, endpoints, _, _, _ = setup(tmp_path)
    with (
        ThreadPoolExecutor(max_workers=1) as pool,
        pytest.raises(RuntimeError, match="owning thread"),
    ):
        pool.submit(rpc.close).result()
    assert not any(x.closed for x in endpoints.values())
    rpc.close()


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True])
def test_stage_duration_rejects_invalid_time(value):
    Latencies, _, _, _ = api()
    with pytest.raises(ValueError):
        Latencies(bootstrap_s=0.1, s1_s=value, s2_s=0.1, recovery_s=0.4)


def test_old_bootstrap_receipt_cannot_run_after_new_bootstrap(tmp_path):
    controller, _, _, old, _, run_old = setup(tmp_path)
    fresh = controller.bootstrap({"frame": -2, "state": {}}, "pick cup", seed=2)
    with pytest.raises(ValueError, match="receipt"):
        run_old(1)
    assert fresh["generation"] > old["generation"]
    assert fresh["receipt_id"] != old["receipt_id"]


def test_failed_bootstrap_invalidates_previous_controller_receipt(tmp_path):
    controller, _, endpoints, _, _, run_old = setup(tmp_path)
    endpoints[1001].mutate = lambda name, result: (
        result.update(forward_counts={"vlm": 0, "dit": 0})
        if name == "bootstrap"
        else None
    )
    with pytest.raises(ValueError, match="forward"):
        controller.bootstrap({"frame": 0, "state": {}}, "pick cup", seed=2)
    with pytest.raises(RuntimeError, match="fresh bootstrap"):
        run_old(1)


def test_external_receipt_actions_and_metadata_cannot_change_private_bootstrap(
    tmp_path,
):
    _, _, _, receipt, steps, run = setup(tmp_path)
    receipt["actions"][:] = 0
    receipt["meta"]["capture_s"] = 800
    receipt["meta"]["source_tick"] = 50
    receipt["seed"] = 999
    result = run(1)
    assert steps[0][1][0] == pytest.approx(0.1)
    assert steps[0][1][6] == -1
    assert result["bootstrap"]["seed"] == 1
    assert result["bootstrap"]["meta"]["capture_s"] == 1000


def test_bootstrap_language_is_bound_to_run(tmp_path):
    _, Step, _, _ = api()
    controller, _, _, receipt, _, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="language"):
        controller.run(
            receipt,
            initial_observation={"frame": 0, "state": {}},
            initial_capture_s=1000.1,
            language="different task",
            step_fn=lambda a, t: Step({}),
            max_steps=1,
        )


@pytest.mark.parametrize("failure", ["rpc", "audit", "actions"])
def test_failed_plan_requires_a_new_successful_bootstrap(tmp_path, failure):
    _, rpc, endpoints, receipt, _, _ = setup(tmp_path)
    feature = {"meta": receipt["meta"], "sequence": 0, "vl_embeds": None}

    def plan(tick=0):
        return rpc.plan(
            {"state": {}},
            "pick cup",
            request_tick=tick,
            delay_ticks=1,
            committed_actions=receipt["actions"][:1],
            state_capture_s=1000.1,
            feature=feature,
        )

    def mutate(name, result):
        if name != "plan":
            return
        if failure == "rpc":
            raise TimeoutError("server may already have changed its buffer")
        if failure == "audit":
            result["audit"]["valid_end_tick"] = 99
        else:
            result["actions"] = np.zeros((0, 7))

    endpoints[1001].mutate = mutate
    with pytest.raises((TimeoutError, ValueError)):
        plan()
    endpoints[1001].mutate = None
    with pytest.raises(ValueError, match="bootstrap"):
        plan(1)
    rpc.bootstrap(
        {"frame": 0, "state": {}}, "pick cup", capture_s=1000, delay_ticks=1, seed=2
    )
    assert plan()["audit"]["request_tick"] == 0


@pytest.mark.parametrize("failure", ["rpc", "audit", "actions"])
def test_failed_rpc_bootstrap_does_not_leave_a_usable_anchor(tmp_path, failure):
    _, rpc, endpoints, receipt, _, _ = setup(tmp_path)

    def mutate(name, result):
        if name != "bootstrap":
            return
        if failure == "rpc":
            raise TimeoutError("bootstrap response was lost")
        if failure == "audit":
            result["forward_counts"] = {"vlm": 0, "dit": 0}
        else:
            result["actions"] = np.zeros((0, 7))

    endpoints[1001].mutate = mutate
    with pytest.raises((TimeoutError, ValueError)):
        rpc.bootstrap({"state": {}}, "pick cup", capture_s=1000, delay_ticks=1, seed=2)
    endpoints[1001].mutate = None
    with pytest.raises(ValueError, match="bootstrap"):
        rpc.plan(
            {"state": {}},
            "pick cup",
            request_tick=0,
            delay_ticks=1,
            committed_actions=receipt["actions"][:1],
            state_capture_s=1000.1,
            feature={"meta": receipt["meta"]},
        )


def test_wrong_thread_plan_cannot_revoke_the_owners_bootstrap(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    _, rpc, _, receipt, _, _ = setup(tmp_path)

    def plan():
        return rpc.plan(
            {"state": {}},
            "pick cup",
            request_tick=0,
            delay_ticks=1,
            committed_actions=receipt["actions"][:1],
            state_capture_s=1000.1,
            feature={"meta": receipt["meta"]},
        )

    with (
        ThreadPoolExecutor(max_workers=1) as pool,
        pytest.raises(RuntimeError, match="owning thread"),
    ):
        pool.submit(plan).result()
    assert plan()["audit"]["request_tick"] == 0


@pytest.mark.parametrize("seconds,ticks", [(0.15, [0, 3, 6]), (0.05, list(range(8)))])
def test_exact_boundary_s2_publishes_before_fresh_offer_then_restarts(
    tmp_path, seconds, ticks
):
    _, _, endpoints, _, _, run = setup(tmp_path, s2=seconds)
    result = run(8)
    visions = [data for endpoint, data in endpoints[1002].calls if endpoint == "vision"]
    assert [data["source_tick"] for data in visions] == ticks
    assert [record["source_tick"] for record in requests(result, "s2")] == ticks


def test_terminal_boundary_s2_completion_does_not_start_unneeded_work(tmp_path):
    _, Step, _, _ = api()
    _, _, endpoints, _, _, run = setup(tmp_path, s2=0.15)
    result = run(
        3,
        lambda action, tick: Step({"frame": tick + 1, "state": {}}, success=tick == 2),
    )
    visions = [data for endpoint, data in endpoints[1002].calls if endpoint == "vision"]
    assert [data["source_tick"] for data in visions] == [0]
    assert not any(record["lane"] == "s2" for record in result["inflight_at_end"])
    assert any(event["kind"] == "camera_dropped" for event in result["events"])


class ServiceClock:
    """A controllable full-adapter clock; RPC records use their own real clock."""

    def __init__(self):
        self.now = 20000.0

    def __call__(self):
        return self.now


def native_setup(
    tmp_path,
    *,
    s1=(0.023,),
    s2=(0.071,),
    bootstrap=0.137,
    fixed=None,
    startup_release_seconds=None,
):
    _, rpc, endpoints, _, _, _ = setup(tmp_path)
    _, Step, Controller, _ = api()
    clock = ServiceClock()
    for name, values in (("bootstrap", (bootstrap,)), ("vision", s2), ("plan", s1)):
        method = getattr(rpc, name)
        durations = iter(values)
        last = values[-1]

        def measured(*args, _method=method, _durations=durations, _last=last, **kwargs):
            duration = next(_durations, _last)
            # Include adapter preprocessing and return handling, independently
            # from the RPC's narrower raw wall receipt.
            clock.now += duration / 3
            result = _method(*args, **kwargs)
            clock.now += duration * 2 / 3
            return result

        setattr(rpc, name, measured)
    controller = Controller(
        rpc,
        timing_mode="native",
        clock=clock,
        fixed_delay=fixed,
        startup_release_seconds=startup_release_seconds,
    )
    receipt = controller.bootstrap({"frame": -1, "state": {}}, "pick cup", seed=1)

    def run(n):
        return controller.run(
            receipt,
            initial_observation={"frame": 0, "state": {}},
            initial_capture_s=1000 + max(bootstrap, startup_release_seconds or 0),
            language="pick cup",
            step_fn=lambda action, tick: Step({"frame": tick + 1, "state": {}}),
            max_steps=n,
        )

    return controller, rpc, endpoints, receipt, clock, run


def test_common_startup_release_keeps_every_native_service_duration(tmp_path):
    _, _, _, receipt, _, run = native_setup(
        tmp_path, s1=(0.023, 0.039), s2=(0.071, 0.083), startup_release_seconds=0.4
    )
    result = run(5)
    assert receipt["logical_duration_s"] == pytest.approx(0.137)
    assert receipt["logical_completed_s"] == pytest.approx(1000.137)
    assert receipt["service_wall"]["duration_s"] == receipt["logical_duration_s"]
    for record in (receipt, result, result["config"]):
        assert record["requested_startup_release_seconds"] == 0.4
        assert record["bootstrap_service_seconds"] == pytest.approx(0.137)
        assert record["startup_barrier_seconds"] == pytest.approx(0.263)
        assert record["startup_seconds"] == 0.4
        assert record["control_origin_s"] == 1000.4
        assert record["startup_alignment_met"] is True
        assert record["startup_policy"] == "common_startup_release_moving_world_hold"
    assert requests(result, "s1")[0]["state_capture_s"] == 1000.4
    assert requests(result, "s1")[0]["feature_capture_s"] == 1000
    assert requests(result, "s1")[0]["feature_age_s"] == pytest.approx(0.4)
    assert requests(result, "s1")[0]["image_delay_ticks"] == 8
    assert requests(result, "s1")[0]["image_delay_embedding_ticks"] == 5
    assert [r["logical_duration_s"] for r in requests(result, "s1")][
        :2
    ] == pytest.approx([0.023, 0.039])
    assert [r["logical_duration_s"] for r in requests(result, "s2")][
        :2
    ] == pytest.approx([0.071, 0.083])


def test_receipt_tampering_cannot_change_private_common_release(tmp_path):
    controller, _, _, receipt, _, run = native_setup(
        tmp_path, startup_release_seconds=0.4
    )
    for key in (
        "bootstrap_service_seconds",
        "startup_barrier_seconds",
        "startup_seconds",
        "control_origin_s",
    ):
        receipt[key] = 999
    receipt["requested_startup_release_seconds"] = 999
    receipt["startup_alignment_met"] = False
    snapshot = controller.bootstrap_evidence
    snapshot["startup_seconds"] = 999
    assert controller.startup_metadata()["startup_seconds"] == 0.4
    result = run(1)
    assert result["control_origin_s"] == 1000.4
    assert result["bootstrap"]["startup_seconds"] == 0.4
    assert result["startup_alignment_met"] is True


def test_late_bootstrap_fails_before_actions_and_retains_native_service(tmp_path):
    controller, _, endpoints, receipt, _, run = native_setup(
        tmp_path, bootstrap=0.41, startup_release_seconds=0.4
    )
    assert receipt["startup_alignment_met"] is False
    assert receipt["startup_seconds"] == pytest.approx(0.41)
    assert receipt["startup_barrier_seconds"] == 0
    assert receipt["service_wall"]["duration_s"] == receipt["logical_duration_s"]
    receipt["startup_alignment_met"] = True
    before = {port: len(endpoint.calls) for port, endpoint in endpoints.items()}
    with pytest.raises(RuntimeError, match="startup release"):
        run(2)
    assert {port: len(endpoint.calls) for port, endpoint in endpoints.items()} == before
    assert controller.logical_time_s == pytest.approx(1000.41)
    with pytest.raises(RuntimeError, match="fresh bootstrap"):
        run(1)


def test_late_rebootstrap_does_not_retain_previous_run_actions(tmp_path):
    controller, rpc, _, _, clock, run = native_setup(
        tmp_path, startup_release_seconds=0.4
    )
    assert run(2)["action_steps"] == 2
    original = rpc.bootstrap

    def late(*args, **kwargs):
        receipt = original(*args, **kwargs)
        clock.now += 0.5
        return receipt

    rpc.bootstrap = late
    receipt = controller.bootstrap({"frame": -1, "state": {}}, "pick cup", seed=1)
    with pytest.raises(RuntimeError, match="startup release"):
        controller.run(
            receipt,
            initial_observation={"frame": 0, "state": {}},
            initial_capture_s=receipt["control_origin_s"],
            language="pick cup",
            step_fn=lambda *_: pytest.fail("late startup cannot step"),
            max_steps=2,
        )
    assert controller.actions == controller.requests == []
    assert controller.timeline is None


def test_bootstrap_exactly_at_common_release_needs_no_extra_wait(tmp_path):
    _, _, _, receipt, _, run = native_setup(
        tmp_path, bootstrap=0.375, startup_release_seconds=0.375
    )
    result = run(1)
    assert receipt["startup_alignment_met"] is True
    assert receipt["startup_barrier_seconds"] == 0
    assert result["physical_seconds"] == pytest.approx(0.425)


@pytest.mark.parametrize("seconds", [0, -0.1, True, float("nan"), float("inf")])
def test_common_startup_release_rejects_invalid_time(tmp_path, seconds):
    _, _, Controller, _ = api()
    with pytest.raises(ValueError, match="startup release"):
        Controller(None, timing_mode="native", startup_release_seconds=seconds)


def test_native_uses_every_complete_adapter_duration_without_alignment(tmp_path):
    _, _, _, receipt, _, run = native_setup(
        tmp_path, s1=(0.023, 0.039, 0.017), s2=(0.071, 0.083, 0.056)
    )
    result = run(8)
    assert receipt["logical_duration_s"] == pytest.approx(0.137)
    assert result["timing_mode"] == "native"
    assert result["timing_domain"] == "measured_request_discrete_event"
    assert result["latency_provenance"] == "measured_per_request"
    assert result["resource_model"] == "independent_stage_service"
    assert result["compute_coupling"] == "serialized_gpu_simulation"
    assert result["stage_latencies"] is None
    assert result["config"]["stage_latencies"] is None
    assert result["config"]["initial_delay_ticks"] == 1
    assert result["startup_seconds"] == receipt["logical_duration_s"]
    assert result["physical_seconds"] == pytest.approx(0.537)
    assert [r["logical_duration_s"] for r in requests(result, "s1")][
        :3
    ] == pytest.approx([0.023, 0.039, 0.017])
    assert [r["logical_duration_s"] for r in requests(result, "s2")][
        :3
    ] == pytest.approx([0.071, 0.083, 0.056])
    for row in [receipt, *result["requests"]]:
        service = row["service_wall"]
        assert service["clock"] == "host_monotonic"
        assert service["boundary"] == "full_adapter_call"
        assert service["duration_s"] == service["completed_s"] - service["started_s"]
        assert row["logical_duration_s"] == service["duration_s"]
        assert row["logical_completed_s"] - row.get(
            "logical_started_s", 1000
        ) == pytest.approx(service["duration_s"])
    assert requests(result, "s1")[0]["wall_duration_s"] != pytest.approx(0.023)


def test_native_adapts_only_after_past_completion_and_expires_late_actions(tmp_path):
    _, _, _, _, _, run = native_setup(tmp_path, s1=(0.08, 0.08, 0.03))
    result = run(8)
    plans = requests(result, "s1")
    assert plans[0]["delay_ticks"] == 1
    assert plans[0]["publication"]["expired_slots"] == 1
    assert plans[0]["budget_decision"]["deadline_miss"] is True
    assert plans[1]["delay_ticks"] == 2
    assert plans[1]["publication"]["installed_slots"] == 2


def test_native_recovery_uses_its_actual_call_duration(tmp_path):
    _, _, _, _, _, run = native_setup(tmp_path, s1=(2.2, 0.327, 0.021), fixed=1)
    result = run(55)
    plans = requests(result, "s1")
    assert plans[1]["normal_or_recovery"] == "recovery5"
    assert plans[1]["server_audit"]["forward_counts"] == {"vlm": 0, "dit": 5}
    assert plans[1]["logical_duration_s"] == pytest.approx(0.327)
    assert plans[1]["logical_duration_s"] == plans[1]["service_wall"]["duration_s"]
    assert result["actions"][40]["fallback"] is True


@pytest.mark.parametrize("duration", [-0.01, float("nan"), float("inf")])
def test_native_rejects_invalid_measured_duration(tmp_path, duration):
    _, rpc, _, _, _, _ = setup(tmp_path)
    _, _, Controller, _ = api()
    readings = iter([100.0, 100.0 + duration])
    controller = Controller(rpc, timing_mode="native", clock=lambda: next(readings))
    with pytest.raises(ValueError, match="monotonic.*finite|finite.*monotonic"):
        controller.bootstrap({"frame": -1, "state": {}}, "pick cup", seed=1)
    assert controller._receipt is None


def test_native_rejects_fixed_stage_latencies(tmp_path):
    _, rpc, _, _, _, _ = setup(tmp_path)
    Latencies, _, Controller, _ = api()
    with pytest.raises(ValueError, match="native.*fixed|Native.*fixed"):
        Controller(rpc, timing_mode="native", latencies=Latencies(0.1, 0.1, 0.1, 0.1))


def test_native_completion_does_not_see_host_computed_future_images(tmp_path):
    _, _, endpoints, _, _, run = native_setup(tmp_path, s2=(0.12,))
    result = run(6)
    visions = [d for name, d in endpoints[1002].calls if name == "vision"]
    assert [d["source_tick"] for d in visions][:3] == [0, 2, 4]
    assert [d["observation"]["frame"] for d in visions][:3] == [0, 2, 4]
    assert any(e["kind"] == "camera_replaced" for e in result["events"])


@pytest.mark.parametrize("stage", ["vision", "plan"])
def test_native_invalid_clock_during_steady_service_retains_completed_evidence(
    tmp_path, stage
):
    controller, rpc, _, receipt, clock, run = native_setup(tmp_path)
    original = getattr(rpc, stage)

    def backwards(*args, **kwargs):
        result = original(*args, **kwargs)
        clock.now -= 1
        return result

    setattr(rpc, stage, backwards)
    with pytest.raises(ValueError, match="Measured monotonic"):
        run(3)
    assert controller._receipt["logical_duration_s"] == receipt["logical_duration_s"]
    assert rpc.records[-1]["status"] == "complete"
    assert len(controller.requests) == (1 if stage == "plan" else 0)
    assert controller.actions == []
