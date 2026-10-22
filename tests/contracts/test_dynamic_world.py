"""Physical timing and read-only object registries, without a simulator or model."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from coexecution.libero_dynamic import world


class FixtureEnv:
    """Native wrapper interface with independently advancing physics/body clocks."""

    def __init__(self, motion="body_translation_zigzag_waypoints"):
        self.dynamic_config = {
            "motion": {"type": motion},
            "carrier": {"enabled": False},
        }
        self.dynamic_controller = SimpleNamespace()
        self.dynamic_multi_target_states = []
        self.dynamic_fixture_body_state = {"elapsed": 1.0}
        self.pending_inference_latency_notified = False
        self.pending_inference_latency_seconds = None
        self.dynamic_step_idx = 50
        self.robot_position = 0.2
        self.free_position = 0.4
        self.fixture_position = 1.0
        self.calls = []
        self.config = {
            "enabled": True,
            "mode": "continuous_physics",
            "substep_dt": 0.01,
            "freeze_robot_during_latency": True,
            "apply_fixed_latency_every_step": False,
        }
        self.sim = SimpleNamespace(
            model=SimpleNamespace(opt=SimpleNamespace(timestep=0.002)),
            data=SimpleNamespace(time=3.0),
            step=self.physics_step,
            forward=lambda: None,
        )

    def _get_latency_config(self):
        return dict(self.config)

    def notify_inference_latency(self, seconds):
        self.calls.append(("notify", seconds))
        self.pending_inference_latency_notified = True
        self.pending_inference_latency_seconds = seconds

    def _consume_inference_latency(self):
        self.calls.append(("consume", self.pending_inference_latency_notified))
        value = self.pending_inference_latency_seconds or 0.0
        self.pending_inference_latency_notified = False
        self.pending_inference_latency_seconds = None
        return value

    def _capture_latency_robot_hold_state(self, config):
        return self.robot_position

    def _restore_latency_robot_hold_state(self, state, *, zero_velocity=True):
        assert zero_velocity
        self.robot_position = state

    def _advance_dynamic_fixture_body(self, config, seconds, *, reason):
        self.calls.append(("fixture", seconds))
        self.fixture_position += seconds
        self.dynamic_fixture_body_state["elapsed"] += seconds
        return {
            "delta_seconds": seconds,
            "elapsed_seconds": self.dynamic_fixture_body_state["elapsed"],
        }

    def physics_step(self):
        dt = self.sim.model.opt.timestep
        self.sim.data.time += dt
        self.robot_position += dt
        self.free_position += 2 * dt


@pytest.mark.parametrize(
    "motion", ["body_yaw_zigzag_waypoints", "body_translation_polynomial_spline"]
)
@pytest.mark.parametrize("seconds", [0, 0.06, 0.1, 0.0613, 0.0003])
def test_fixture_hold_moves_real_physics_and_free_objects_for_exact_requested_duration(
    motion, seconds
):
    env = FixtureEnv(motion)
    timing = world.NativeFixtureTiming(env)
    observations = []
    advanced = timing.advance(
        seconds,
        on_substep=lambda value: observations.append((value, env.robot_position)),
    )
    assert advanced == pytest.approx(seconds, abs=1e-10)
    assert env.sim.data.time - 3.0 == pytest.approx(seconds, abs=1e-10)
    assert env.dynamic_fixture_body_state["elapsed"] - 1.0 == pytest.approx(seconds)
    assert env.free_position == pytest.approx(0.4 + 2 * seconds)
    assert env.robot_position == 0.2
    assert all(position == 0.2 for _, position in observations)
    assert env.dynamic_step_idx == 50
    assert env.sim.model.opt.timestep == 0.002
    assert env._consume_inference_latency() == 0  # The next env.step cannot hold again.
    if seconds:
        assert observations[-1][0] == pytest.approx(seconds)
    record = timing.records[-1]
    assert record["latency_backend"] == "owned_fixture_exact_hold"
    assert record["physics_elapsed_s"] == pytest.approx(seconds)
    assert record["source_native_duration_s"] == pytest.approx(
        np.ceil(seconds / 0.01) * 0.01
    )


def native_hold_reference(env, seconds):
    """Pinned _apply_latency_drift/_advance_latency_physics timing for body drivers."""
    for _ in range(int(np.ceil(seconds / env.config["substep_dt"]))):
        hold = env._capture_latency_robot_hold_state(env.config)
        env._advance_dynamic_fixture_body(env.config, 0.01, reason="source")
        for _ in range(5):
            env._restore_latency_robot_hold_state(hold)
            env.sim.step()
            env._restore_latency_robot_hold_state(hold)


@pytest.mark.parametrize("seconds", [0.01, 0.06, 0.1, 0.25])
def test_exact_adapter_matches_source_fixture_and_physics_when_latency_is_10ms_multiple(
    seconds,
):
    source, owned = FixtureEnv(), FixtureEnv()
    native_hold_reference(source, seconds)
    world.NativeFixtureTiming(owned).advance(seconds)
    assert owned.fixture_position == pytest.approx(source.fixture_position)
    assert owned.free_position == pytest.approx(source.free_position)
    assert owned.robot_position == source.robot_position
    assert owned.sim.data.time == pytest.approx(source.sim.data.time)


def test_nonmultiple_explicitly_differs_from_published_native_rounding():
    source, owned = FixtureEnv(), FixtureEnv()
    native_hold_reference(source, 0.0613)
    timing = world.NativeFixtureTiming(owned)
    timing.advance(0.0613)
    assert source.sim.data.time - owned.sim.data.time == pytest.approx(0.0087)
    assert timing.records[-1]["source_native_duration_s"] == pytest.approx(0.07)
    assert timing.records[-1]["requested_duration_s"] == 0.0613


def test_hold_failure_restores_timestep_and_robot_and_retains_partial_time():
    env = FixtureEnv()
    timing = world.NativeFixtureTiming(env)

    def fail(_):
        raise RuntimeError("recording failed")

    with pytest.raises(RuntimeError, match="recording failed"):
        timing.advance(0.001, on_substep=fail)
    assert env.sim.model.opt.timestep == 0.002
    assert env.robot_position == 0.2
    assert timing.records[-1]["status"] == "error"
    assert timing.records[-1]["physics_elapsed_s"] == pytest.approx(0.001)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda env: env.dynamic_config["carrier"].update(enabled=True),
        lambda env: env.dynamic_config["motion"].update(type="joint_sine"),
        lambda env: env.config.update(mode="discrete_shift"),
        lambda env: env.config.update(apply_fixed_latency_every_step=True),
        lambda env: env.config.update(freeze_joint_names=["drawer_joint"]),
    ],
)
def test_fixture_adapter_rejects_other_drivers_and_unreviewed_hold_contracts(mutate):
    env = FixtureEnv()
    mutate(env)
    with pytest.raises(ValueError):
        world.NativeFixtureTiming(env)


def test_registry_recurses_coupled_and_passenger_relations_without_changing_drivers():
    target = {"target_object_name": "support", "coupled_object_names": ["carried"]}
    task = {
        "target_objects": [
            target,
            {"target_object_name": "carried", "coupled_object_names": ["nested"]},
        ]
    }
    before = copy.deepcopy(task)
    obj = lambda name: SimpleNamespace(
        joints=[name + "_joint"], root_body=name + "_root"
    )
    states = [
        {
            "name": "passenger",
            "joint": "passenger_joint",
            "dynamic_role": "fixture_supported_passenger_conveyor",
        }
    ]
    env = SimpleNamespace(
        env=SimpleNamespace(
            objects_dict={
                name: obj(name)
                for name in ["support", "carried", "nested", "unrelated", "passenger"]
            },
            fixtures_dict={},
            obj_of_interest=["carried"],
        ),
        dynamic_multi_target_states=states,
    )
    registry = world.observation_registry({"task_cfg": task}, env, [target])
    by_name = {row["target_object_name"]: row for row in registry}
    assert set(by_name) == {"support", "carried", "nested", "unrelated", "passenger"}
    assert "coupled_object" in by_name["nested"]["observation_roles"]
    assert "passenger" in by_name["passenger"]["observation_roles"]
    assert by_name["carried"]["target_joint_name"] == "carried_joint"
    assert by_name["carried"]["target_body_name"] == "carried_root"
    assert task == before
    assert env.dynamic_multi_target_states is states and len(states) == 1
    assert len(target) == 2  # No registry fields were injected into source config.


def test_registry_prefers_fixture_root_over_actuated_child_and_reads_velocity():
    fixture = SimpleNamespace(joints=["drawer_joint"], root_body="cabinet_root")
    env = SimpleNamespace(
        env=SimpleNamespace(
            objects_dict={},
            fixtures_dict={"cabinet": fixture},
            obj_of_interest=["cabinet"],
        ),
        sim=SimpleNamespace(
            data=SimpleNamespace(
                get_joint_qpos=lambda name: np.array([0.2]),
                get_joint_qvel=lambda name: np.array([0.3]),
            )
        ),
    )
    rows = world.observation_registry(
        {
            "task_cfg": {
                "target_object_name": "cabinet",
                "target_joint_name": "drawer_joint",
            }
        },
        env,
        [],
    )
    assert rows[0]["target_body_name"] == "cabinet_root"
    seen = []

    def source_pose(env, name, entry):
        seen.append(entry["target_body_name"])
        return {
            "name": name,
            "joint_name": entry["target_joint_name"],
            "qpos": [0.2],
            "body_name": entry["target_body_name"],
        }

    poses = world.observe_objects(env, rows, source_pose)
    assert seen == ["cabinet_root"]
    assert poses["cabinet"]["qvel"] == [0.3]
    assert poses["cabinet"]["joint_states"]["drawer_joint"] == {
        "qpos": [0.2],
        "qvel": [0.3],
    }


def test_invalid_or_clipped_hold_never_leaves_a_pending_notification():
    env = FixtureEnv()
    timing = world.NativeFixtureTiming(env)
    for seconds in (-1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            timing.advance(seconds)
    assert not env.calls
    consume = env._consume_inference_latency
    env._consume_inference_latency = lambda: min(consume(), 0.05)
    with pytest.raises(ValueError, match="clipped"):
        timing.advance(0.1)
    assert env.pending_inference_latency_notified is False
    assert env.pending_inference_latency_seconds is None
    assert env.sim.data.time == 3.0


def test_registry_records_platform_joints_separately_from_free_object():
    platform = {
        "body_name": "platform_body",
        "joint_names": {"x": "px", "y": "py", "z": "pz"},
    }
    states = [{"name": "cup", "joint": "cup_freejoint", "platform": platform}]
    env = SimpleNamespace(dynamic_multi_target_states=states)
    rows = world.observation_registry({}, env, [])
    cup, support = rows
    assert cup["target_joint_name"] == "cup_freejoint"
    assert support["observation_joint_names"] == ["px", "py", "pz"]
    assert support["observation_roles"] == ["driven_support"]
    support["observation_joint_names"].append("local_only")
    assert len(platform["joint_names"]) == 3
    assert states[0]["platform"] is platform


def test_source_runtime_passenger_role_is_preserved_without_copying_driver_config():
    source = {
        "target_object_name": "passenger",
        "dynamic_role": "fixture_supported_passenger_conveyor",
        "motion": {"waypoints": [[1, 2, 3]]},
    }
    rows = world.observation_registry({}, SimpleNamespace(), [source])
    assert rows[0]["observation_roles"] == ["source_target", "passenger"]
    assert "motion" not in rows[0]
    assert "motion" in source


@pytest.mark.parametrize(
    "motion_type",
    ["body_translation_zigzag_waypoints", "body_translation_polynomial_spline"],
)
def test_goal7_resolved_latency_body_type_overrides_inherited_conveyor_trajectory(
    motion_type,
):
    # Goal task 7 overrides latency.type but leaves defaults.latency.trajectory
    # intact in every reviewed L1/L2/L3 resolved config. Source's
    # _resolved_motion_type prioritizes body_* type over the inherited string.
    env = FixtureEnv(motion_type)
    env.config.update(
        type=motion_type,
        trajectory="conveyor_pingpong",
        start_step=30,
        end_step=1_000_000,
        dt_seconds=0.05,
        alpha_mps=0.055,
    )
    timing = world.NativeFixtureTiming(env)
    assert timing.motion_config["type"] == motion_type
    assert timing.motion_config["trajectory"] == "conveyor_pingpong"
    assert timing.advance(0.0613) == pytest.approx(0.0613)


def test_conflicting_effective_body_latency_type_is_still_rejected():
    env = FixtureEnv("body_translation_zigzag_waypoints")
    env.config.update(
        type="body_yaw_zigzag_waypoints", trajectory="body_translation_zigzag_waypoints"
    )
    with pytest.raises(ValueError, match="driver differs"):
        world.NativeFixtureTiming(env)


def test_zero_latency_consumes_notification_without_touching_physical_state():
    env = FixtureEnv()
    env.sim.data.qpos = np.array([0.2, 0.4])
    env.sim.data.qvel = np.array([0.7, 0.9])
    env.sim.model.body_pos = np.array([[1.0, 2.0, 3.0]])
    before = (
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        env.sim.model.body_pos.copy(),
    )
    calls = []
    env._capture_latency_robot_hold_state = lambda config: calls.append("capture")

    def restore(state, **kwargs):
        calls.append("restore")
        env.sim.data.qvel[0] = 0

    env._restore_latency_robot_hold_state = restore
    env.sim.forward = lambda: calls.append("forward")
    timing = world.NativeFixtureTiming(env)
    assert timing.advance(0, on_substep=lambda value: calls.append("record")) == 0
    assert calls == []
    for current, prior in zip(
        (env.sim.data.qpos, env.sim.data.qvel, env.sim.model.body_pos), before
    ):
        np.testing.assert_array_equal(current, prior)
    assert env.sim.data.time == 3.0
    assert env.dynamic_fixture_body_state["elapsed"] == 1.0
    assert env.pending_inference_latency_notified is False
    assert env.pending_inference_latency_seconds is None
    assert timing.records[-1]["status"] == "complete"
    assert timing.records[-1]["physics_elapsed_s"] == 0
    assert timing.records[-1]["fixture_elapsed_s"] == 0


def test_yaw_speed_evidence_preserves_angular_speed_independent_of_nominal_linear_speed():
    env = FixtureEnv("body_yaw_zigzag_waypoints")
    env.dynamic_config["motion"].update(alpha_mps=0.09, alpha_yaw_rad_per_s=0.18)
    env.config.update(
        type="body_yaw_zigzag_waypoints", alpha_mps=0.09, alpha_yaw_rad_per_s=0.18
    )
    evidence = world.NativeFixtureTiming(env).driver_evidence
    assert evidence["speed_parameter"] == "alpha_yaw_rad_per_s"
    assert evidence["speed_unit"] == "rad/s"
    assert evidence["action_speed"] == evidence["latency_speed"] == 0.18
    assert evidence["action_motion"]["alpha_mps"] == 0.09
    assert evidence["latency_motion"]["alpha_mps"] == 0.09
