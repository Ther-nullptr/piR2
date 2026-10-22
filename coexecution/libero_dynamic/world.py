"""Owned fixture timing and read-only scene observations.

The pinned source owns fixture trajectories, robot hold selection, initialization
and control actions. Its non-preview latency loop advances whole 10 ms blocks.
This adapter exposes that physical hold separately and shortens only its final
block to the requested duration. No free object or task joint is held by it.
"""

from __future__ import annotations

import copy
import math

import numpy as np


class NativeFixtureTiming:
    """Exact synchronous hold for the source's no-carrier fixture-body route."""

    backend = "owned_fixture_exact_hold"

    def __init__(self, env):
        self.env = env
        self.records = []
        config = env.dynamic_config
        motion = config.get("motion") or {}
        self.latency_config = copy.deepcopy(env._get_latency_config())
        latency = self.latency_config
        motion_type = str(motion.get("type", "")).lower()
        if (
            config.get("carrier", {}).get("enabled", True)
            or not motion_type.startswith(("body_yaw_", "body_translation_"))
            or env.dynamic_controller is None
            or env.dynamic_multi_target_states
            or env.dynamic_fixture_body_state is None
        ):
            raise ValueError(
                "Exact fixture hold requires one native no-carrier body driver"
            )
        if (
            not latency.get("enabled", False)
            or latency.get("mode") != "continuous_physics"
            or not latency.get("freeze_robot_during_latency", True)
            or latency.get("apply_fixed_latency_every_step", False)
            or latency.get("freeze_joint_names") is not None
            or tuple(latency.get("freeze_joint_prefixes", ["robot0_", "gripper0_"]))
            != ("robot0_", "gripper0_")
        ):
            raise ValueError(
                "Fixture hold requires notified latency and robot-only freezing"
            )
        self.block_dt = float(latency.get("substep_dt", 0))
        self.physics_dt = float(env.sim.model.opt.timestep)
        if (
            not math.isfinite(self.block_dt)
            or not math.isclose(self.block_dt, 0.01, rel_tol=0, abs_tol=1e-12)
            or not math.isfinite(self.physics_dt)
            or self.physics_dt <= 0
            or not math.isclose(
                self.block_dt / self.physics_dt,
                round(self.block_dt / self.physics_dt),
                abs_tol=1e-10,
            )
        ):
            raise ValueError(
                "Reviewed fixture hold requires integral physics steps per 10 ms block"
            )
        # Source configs put trajectory parameters in both motion and latency.
        # Missing latency parameters may inherit motion, but an explicit different
        # driver is not silently substituted.
        self.motion_config = {**copy.deepcopy(motion), **latency}
        # Match pinned _resolved_motion_type: a body_* type takes precedence
        # over trajectory, which may still be the inherited conveyor default.
        declared = str(self.motion_config.get("type", "")).lower()
        if not declared.startswith(("body_yaw_", "body_translation_")):
            declared = str(self.motion_config.get("trajectory", "")).lower() or declared
        if declared != motion_type:
            raise ValueError("Fixture latency driver differs from action motion")
        speed_parameter = (
            "alpha_yaw_rad_per_s"
            if motion_type.startswith("body_yaw_")
            else "alpha_mps"
        )
        self.driver_evidence = {
            "effective_motion_type": declared,
            "speed_parameter": speed_parameter,
            "speed_unit": "rad/s" if motion_type.startswith("body_yaw_") else "m/s",
            "action_speed": motion.get(speed_parameter),
            "latency_speed": self.motion_config.get(speed_parameter),
            "action_motion": copy.deepcopy(motion),
            "latency_motion": copy.deepcopy(self.motion_config),
        }

    def advance(self, seconds, *, on_substep=None):
        """Advance actual physics while keeping only source-selected robot joints fixed."""
        if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Fixture hold duration must be finite nonnegative seconds")
        seconds = float(seconds)
        env, sim = self.env, self.env.sim
        start_time = float(sim.data.time)
        start_fixture_time = float(env.dynamic_fixture_body_state["elapsed"])
        record = {
            "latency_backend": self.backend,
            "requested_duration_s": seconds,
            "source_substep_dt_s": self.block_dt,
            "source_native_duration_s": math.ceil(seconds / self.block_dt)
            * self.block_dt,
            "source_rounding_rule": "ceil(tau / latency.substep_dt) whole blocks",
            "physics_started_s": start_time,
            "fixture_started_s": start_fixture_time,
            "fixture_events": [],
            "status": "error",
        }
        self.records.append(record)
        # Notify then consume once using the source contract. This retains its
        # measurement evidence and ensures the following native env.step is clean.
        env.notify_inference_latency(seconds)
        consumed = float(env._consume_inference_latency())
        record["consumed_latency_s"] = consumed
        if not math.isclose(consumed, seconds, rel_tol=0, abs_tol=1e-12):
            record.update(physics_elapsed_s=0.0, fixture_elapsed_s=0.0)
            raise ValueError(
                "Native latency selection clipped or changed requested duration"
            )
        if seconds == 0:
            # Native _apply_latency_drift returns before freezing joints or
            # forwarding physics for zero latency. Preserve robot qvel as well.
            record.update(
                status="complete",
                physics_elapsed_s=0.0,
                fixture_elapsed_s=0.0,
                physics_completed_s=start_time,
                fixture_completed_s=start_fixture_time,
            )
            return 0.0
        hold = env._capture_latency_robot_hold_state(self.latency_config)
        original_dt = float(sim.model.opt.timestep)
        elapsed = 0.0
        try:
            env._restore_latency_robot_hold_state(hold, zero_velocity=True)
            while elapsed < seconds - 1e-12:
                block = min(self.block_dt, seconds - elapsed)
                record["fixture_events"].append(
                    env._advance_dynamic_fixture_body(
                        self.motion_config, block, reason="owned_fixture_exact_hold"
                    )
                )
                block_elapsed = 0.0
                while block_elapsed < block - 1e-12:
                    dt = min(original_dt, block - block_elapsed)
                    sim.model.opt.timestep = dt
                    env._restore_latency_robot_hold_state(hold, zero_velocity=True)
                    sim.step()
                    env._restore_latency_robot_hold_state(hold, zero_velocity=True)
                    block_elapsed += dt
                    elapsed += dt
                    # Forward after restoring joints, before rendering or reading
                    # their world poses; never publish a temporary drifting robot.
                    sim.forward()
                    if on_substep is not None:
                        on_substep(float(sim.data.time) - start_time)
            advanced = float(sim.data.time) - start_time
            if not math.isclose(advanced, seconds, rel_tol=0, abs_tol=1e-8):
                raise RuntimeError(
                    "Fixture hold physics clock differs from requested duration"
                )
            record["status"] = "complete"
            return advanced
        finally:
            sim.model.opt.timestep = original_dt
            env._restore_latency_robot_hold_state(hold, zero_velocity=True)
            sim.forward()
            record["physics_elapsed_s"] = float(sim.data.time) - start_time
            record["fixture_elapsed_s"] = (
                float(env.dynamic_fixture_body_state["elapsed"]) - start_fixture_time
            )
            record["physics_completed_s"] = float(sim.data.time)
            record["fixture_completed_s"] = float(
                env.dynamic_fixture_body_state["elapsed"]
            )


def _layers(env):
    seen = set()
    while env is not None and id(env) not in seen:
        seen.add(id(env))
        yield env
        env = getattr(env, "env", None)


def _names(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, dict):
        return list(value.values())
    return list(value)


def observation_registry(item, env, source_entries):
    """Resolve observation-only objects, fixtures and support bodies, without writes.

    Entries are independent copies. Adding a passenger/coupled object here never
    installs a driver, changes a target configuration, or freezes its free joint.
    """
    entries = {}

    def add(entry, role):
        if not isinstance(entry, dict):
            return
        name = (
            entry.get("target_object_name")
            or entry.get("object_name")
            or entry.get("name")
        )
        if not name:
            return
        name = str(name)
        row = entries.setdefault(
            name, {"target_object_name": name, "observation_roles": []}
        )
        roles = row["observation_roles"]
        if role not in roles:
            roles.append(role)
        if (
            "passenger" in str(entry.get("dynamic_role", ""))
            and "passenger" not in roles
        ):
            roles.append("passenger")
        # Retain pose resolution only, not whole trajectory configs/activation
        # variants or live platform state in this observation-only registry.
        for key in (
            "target_joint_name",
            "joint_name",
            "target_body_name",
            "body_name",
            "coupled_object_names",
            "dynamic_role",
            "observation_joint_names",
        ):
            if key in entry:
                row.setdefault(key, copy.deepcopy(entry[key]))
        if entry.get("joint"):
            row.setdefault("target_joint_name", str(entry["joint"]))
        for child in _names(entry.get("coupled_object_names")):
            add({"target_object_name": str(child)}, "coupled_object")

    for entry in source_entries:
        add(entry, "source_target")
    task = item.get("task_cfg") or {}
    add(task, "source_target")
    for entry in task.get("target_objects") or []:
        add(entry, "source_target")
    for layer in _layers(env):
        for name, obj in (getattr(layer, "objects_dict", {}) or {}).items():
            add({"target_object_name": name}, "scene_object")
            row = entries[str(name)]
            joints = _names(getattr(obj, "joints", []))
            row["observation_joint_names"] = list(
                dict.fromkeys([*row.get("observation_joint_names", []), *joints])
            )
            if joints:
                row.setdefault("target_joint_name", joints[0])
            root = getattr(obj, "root_body", None)
            if root:
                row["target_body_name"] = str(root)
        for name, fixture in (getattr(layer, "fixtures_dict", {}) or {}).items():
            add({"target_object_name": name}, "fixture")
            row = entries[str(name)]
            joints = _names(getattr(fixture, "joints", []))
            row["observation_joint_names"] = list(
                dict.fromkeys([*row.get("observation_joint_names", []), *joints])
            )
            if joints:
                row.setdefault("target_joint_name", joints[0])
            root = getattr(fixture, "root_body", None)
            if root:
                row["target_body_name"] = str(root)
        for name in getattr(layer, "obj_of_interest", []) or []:
            add({"target_object_name": name}, "task_object_of_interest")
    for state in getattr(env, "dynamic_multi_target_states", []) or []:
        role = (
            "passenger"
            if "passenger" in str(state.get("dynamic_role", ""))
            else "runtime_target"
        )
        add(state, role)
        platform = state.get("platform") or {}
        if platform.get("body_name"):
            joints = _names(platform.get("joint_names"))
            add(
                {
                    "target_object_name": platform["body_name"],
                    "target_body_name": platform["body_name"],
                    "observation_joint_names": joints,
                },
                "driven_support",
            )
    fixture = getattr(env, "dynamic_fixture_body_state", None) or {}
    if fixture.get("target_name"):
        add({"target_object_name": fixture["target_name"]}, "fixture_driver")
        entries[str(fixture["target_name"])]["target_body_name"] = fixture.get(
            "body_name"
        )
    return list(entries.values())


def observe_objects(env, registry, source_pose):
    """Read world poses plus every known object/fixture joint position and velocity."""
    poses = {}
    for entry in registry:
        name = entry["target_object_name"]
        pose = source_pose(env, name, entry)
        pose["observation_roles"] = list(entry["observation_roles"])
        joint_names = _names(entry.get("observation_joint_names"))
        primary = pose.get("joint_name") or entry.get("target_joint_name")
        if primary:
            joint_names = list(dict.fromkeys([primary, *joint_names]))
        states = {}
        for joint in joint_names:
            state = {}
            for field in ("qpos", "qvel"):
                try:
                    state[field] = np.asarray(
                        getattr(env.sim.data, f"get_joint_{field}")(joint)
                    ).tolist()
                except (KeyError, ValueError, AttributeError):
                    state[field] = None
            states[joint] = state
        pose["joint_states"] = states
        pose["qvel"] = states.get(primary, {}).get("qvel")
        poses[name] = pose
    return poses
