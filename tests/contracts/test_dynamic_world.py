"""Physical timing and read-only object registries, without a simulator or model."""

import ast
import copy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from coexecution.libero_dynamic import world


def test_scene_drive_metadata_defaults_and_rejects_unknown_control():
    assert world.scene_drive_metadata(SimpleNamespace()) == {
        "scene_drive": "moving",
        "scene_drive_scope": "source_driver",
    }
    for value in ("static", None, True, []):
        with pytest.raises(ValueError, match="scene_drive"):
            world.scene_drive_metadata(SimpleNamespace(scene_drive=value))


class CarrierEnv:
    """Joint-addressed host double; every physical write is observable."""

    def __init__(self):
        bodies = {"pot0_main": 1, "pot1_main": 2, "robot0_link": 5}
        names = [
            "pot0_joint",
            *[f"libero_dynamic_preview_platform_0_pot0_{a}" for a in "xyz"],
            "pot1_joint",
            *[f"libero_dynamic_preview_platform_1_pot1_{a}" for a in "xyz"],
            "robot0_joint",
        ]
        bodies.update(
            {
                "libero_dynamic_preview_platform_0_pot0": 3,
                "libero_dynamic_preview_platform_1_pot1": 4,
            }
        )
        model = SimpleNamespace(
            joint_name2id=names.index,
            body_name2id=bodies.__getitem__,
            jnt_type=np.array([0, 2, 2, 2, 0, 2, 2, 2, 3]),
            jnt_bodyid=np.array([1, 3, 3, 3, 2, 4, 4, 4, 5]),
            jnt_qposadr=np.array([0, 7, 8, 9, 10, 17, 18, 19, 20]),
            jnt_dofadr=np.array([0, 6, 7, 8, 9, 15, 16, 17, 18]),
        )
        qpos = np.array(
            [
                0.1,
                0.2,
                0.9,
                1,
                0,
                0,
                0,
                0.06,
                0.18,
                0.83,
                0.5,
                0.2,
                0.9,
                1,
                0,
                0,
                0,
                0.46,
                0.18,
                0.83,
                0.7,
            ]
        )
        qvel = np.array(
            [
                0.001,
                -0.002,
                0,
                0.03,
                0,
                0,
                0.001,
                -0.002,
                0,
                0.001,
                -0.002,
                0,
                0.03,
                0,
                0,
                0.001,
                -0.002,
                0,
                0.2,
            ]
        )
        self.writes = []
        data = SimpleNamespace(time=4.5, qpos=qpos, qvel=qvel)

        def read(name, velocity=False):
            index = names.index(name)
            address = (model.jnt_dofadr if velocity else model.jnt_qposadr)[index]
            count = (6 if velocity else 7) if model.jnt_type[index] == 0 else 1
            values = data.qvel if velocity else data.qpos
            return (
                values[address : address + count]
                if count > 1
                else float(values[address])
            )

        def write(name, value, velocity=False):
            index = names.index(name)
            address = (model.jnt_dofadr if velocity else model.jnt_qposadr)[index]
            values = data.qvel if velocity else data.qpos
            flat = np.asarray(value).reshape(-1)
            values[address : address + len(flat)] = flat
            self.writes.append((name, "qvel" if velocity else "qpos"))

        data.get_joint_qpos = lambda name: read(name)
        data.get_joint_qvel = lambda name: read(name, True)
        data.set_joint_qpos = lambda name, value: write(name, value)
        data.set_joint_qvel = lambda name, value: write(name, value, True)
        self.sim = SimpleNamespace(model=model, data=data, forward=lambda: None)

    def states(self):
        return [
            {
                "name": f"pot{i}",
                "joint": f"pot{i}_joint",
                "target_body_name": f"pot{i}_main",
                "is_joint": False,
                "is_body_yaw": False,
                "is_body_translation": False,
                "carrier_type": "physical_conveyor_patch",
                "platform": {
                    "body_name": f"libero_dynamic_preview_platform_{i}_pot{i}",
                    "kinematic_mode": "slide_joints",
                    "joint_names": {
                        a: f"libero_dynamic_preview_platform_{i}_pot{i}_{a}"
                        for a in "xyz"
                    },
                },
                "motion": {
                    "type": "zigzag_waypoints",
                    "waypoints_xyz": [[0, 0, 0], [0.1, 0.2, 0]],
                    "alpha_mps": 0.001,
                    "loop": True,
                },
                "base_qpos": np.array([0, 0, 0.8, 1, 0, 0, 0]),
                "platform_base_pos": np.zeros(3),
                "object_base_xy": np.zeros(2),
                "object_to_platform_xy": np.zeros(2),
                "contact_events": 0,
                "collision_keys": set(),
            }
            for i in range(2)
        ]


def test_frozen_drive_captures_actual_anchors_without_physical_mutation():
    env = CarrierEnv()
    states = env.states()
    original = copy.deepcopy(states)
    qpos, qvel = env.sim.data.qpos.copy(), env.sim.data.qvel.copy()
    frozen, receipt = world.freeze_preview_states(env, states, phase_s=2.0)
    assert env.writes == []
    np.testing.assert_array_equal(env.sim.data.qpos, qpos)
    np.testing.assert_array_equal(env.sim.data.qvel, qvel)
    assert env.sim.data.time == 4.5
    assert receipt["capture_no_state_mutation"] is True
    assert receipt["capture_state_before"] == receipt["capture_state_after"]
    assert receipt["phase_s"] == 2.0 and receipt["physics_time_s"] == 4.5
    for index, state in enumerate(frozen):
        carrier = receipt["carriers"][index]
        np.testing.assert_array_equal(
            state["base_qpos"], env.sim.data.get_joint_qpos(state["joint"])
        )
        np.testing.assert_array_equal(
            state["platform_base_pos"], [0.06 + 0.4 * index, 0.18, 0.83]
        )
        assert state["motion"]["waypoints_xyz"] == [[0.0, 0.0, 0.0]]
        assert state["motion"]["alpha_mps"] == 0.001
        assert (
            carrier["target_qvel"]
            == env.sim.data.get_joint_qvel(state["joint"]).tolist()
        )
        assert carrier["original_motion"] == original[index]["motion"]
    frozen[0]["motion"]["waypoints_xyz"][0][0] = 10
    assert states[0]["motion"] == original[0]["motion"]
    np.testing.assert_array_equal(states[0]["base_qpos"], original[0]["base_qpos"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s, e: s[1].update(is_joint=True),
        lambda s, e: s[1].update(is_body_yaw=True),
        lambda s, e: s[1].update(carrier_type="direct_qpos"),
        lambda s, e: s[1].update(carrier_type="surface_velocity"),
        lambda s, e: s[1]["platform"].update(kinematic_mode="mocap"),
        lambda s, e: s[1]["platform"].update(kinematic_mode="freejoint"),
        lambda s, e: s[1]["platform"].update(body_name="robot0_link"),
        lambda s, e: s[1]["platform"]["joint_names"].update(x="robot0_joint"),
        lambda s, e: s[1]["motion"].update(alpha_mps=0),
        lambda s, e: e.sim.model.jnt_bodyid.__setitem__(5, 1),
        lambda s, e: e.sim.model.jnt_type.__setitem__(5, 3),
        lambda s, e: s.__setitem__(1, copy.deepcopy(s[0])),
    ],
)
def test_frozen_drive_rejects_mixed_unsupported_states_atomically(mutate):
    env = CarrierEnv()
    states = env.states()
    mutate(states, env)
    old_motion = copy.deepcopy(states[0]["motion"])
    qpos, qvel = env.sim.data.qpos.copy(), env.sim.data.qvel.copy()
    with pytest.raises(ValueError):
        world.freeze_preview_states(env, states, phase_s=2.0)
    assert env.writes == [] and states[0]["motion"] == old_motion
    np.testing.assert_array_equal(env.sim.data.qpos, qpos)
    np.testing.assert_array_equal(env.sim.data.qvel, qvel)


def test_constant_drive_uses_pinned_source_without_constraining_free_targets():
    path = (
        Path(os.environ.get("LIBERO_DYNAMIC_SOURCE", "third_party/libero_dynamic"))
        / "scripts/render_dynamic_config_previews.py"
    )
    if not path.is_file():
        pytest.skip("Pinned dynamic source is not installed; set LIBERO_DYNAMIC_SOURCE")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "25eff7b0f20d77064196b9b9e597b68d3013d35d9a9b78f0659285c2da4ceb76"
    )
    names = {
        "polyline_offset",
        "movable_offset",
        "motion_velocity",
        "set_platform_state",
        "move_physical_platform_preview",
    }
    functions = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {f.name for f in functions} == names
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *functions,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(tree)
    scope = {"np": np, "has_platform_contact": lambda *a: (False, [])}
    exec(compile(tree, str(path), "exec"), scope)  # noqa: S102 -- hash-pinned driver functions, without module imports
    env = CarrierEnv()
    frozen, _ = world.freeze_preview_states(env, env.states(), phase_s=2.0)
    # A robot has now displaced the free object. The external-drive adapter must
    # leave it there, not drag it back to the recorded anchor.
    env.sim.data.qpos[0] += 0.2
    moved = env.sim.data.get_joint_qpos("pot0_joint").copy()
    for elapsed in [2, 2.001, 5, 1000]:
        np.testing.assert_array_equal(
            scope["movable_offset"](frozen[0]["motion"], elapsed), np.zeros(3)
        )
        np.testing.assert_array_equal(
            scope["motion_velocity"](frozen[0]["motion"], elapsed, 0.002), np.zeros(3)
        )
        scope["move_physical_platform_preview"](env, frozen[0], elapsed, 0.002)
        np.testing.assert_array_equal(env.sim.data.get_joint_qpos("pot0_joint"), moved)
        assert all(
            name.startswith("libero_dynamic_preview_platform_")
            for name, _ in env.writes
        )


@pytest.mark.parametrize("presteps", [39, 40])
def test_shared_episode_freezes_only_after_complete_source_setup(presteps):
    from coexecution.libero_dynamic.physical import make_physical_episode_type

    env = CarrierEnv()
    original = env.states()

    class Base:
        def __init__(self, item, args, worker_slot):
            self.item = item
            self.use_preview = True
            self.preview_elapsed = 2.0
            self.preview_states = original

        def _setup_world(self, passed_env, initial):
            assert passed_env is env
            assert self.preview_states is original
            assert len(self.preview_states[0]["motion"]["waypoints_xyz"]) == 2
            return {"raw": "unchanged"}, 80, 10, presteps, []

    source = SimpleNamespace(
        runtime=SimpleNamespace(
            RealtimeEpisode=Base,
            _target_entries=lambda *a: [],
            _object_pose=lambda env, name, entry: {"name": name},
        )
    )
    options = SimpleNamespace(scene_drive="frozen", video_fps=60, save_videos=False)
    episode = make_physical_episode_type(source, options)({}, SimpleNamespace())
    if presteps != 40:
        with pytest.raises(ValueError, match="40"):
            episode._setup_world(env, None)
        assert episode.drive_freeze is None
        assert episode.preview_states is original
    else:
        result = episode._setup_world(env, None)
        assert result[0] == {"raw": "unchanged"}
        assert episode.preview_states is not original
        assert env.dynamic_multi_target_states is episode.preview_states
        assert episode.drive_freeze["phase_s"] == 2.0
        assert episode.drive_freeze["capture_no_state_mutation"]
        assert all(e["scene_drive"] == "frozen" for e in episode.motion_events)
        assert any(e["kind"] == "drive_frozen" for e in episode.motion_events)
    assert env.writes == []


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
