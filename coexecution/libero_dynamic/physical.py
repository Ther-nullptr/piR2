"""Shared moving-world hooks; no policy/model imports or simulator construction."""

import math
import time

import numpy as np

from .world import NativeFixtureTiming, observation_registry, observe_objects


def make_physical_episode_type(source, options, *, callback_convention="source_run"):
    """Own physical evidence while preserving the pinned initialization/drivers.

    source_run compensates the upstream run() callback's subtraction of pending
    fixture latency. physical_origin is for owned loops whose callback starts at
    the actual current world time, and must never receive that compensation.
    """
    if callback_convention not in ("source_run", "physical_origin"):
        raise ValueError("Unknown physical callback convention")
    runtime = source.runtime

    class PhysicalEpisode(runtime.RealtimeEpisode):
        def __init__(self, item, args):
            super().__init__(item, args, worker_slot=0)
            self.env = None
            self.initial_observation = None
            self.last_capture_s = None
            self.control_started = False
            self.rollout_sim_time_s = 0.0
            self.motion_events = []
            self.action_step = 0
            self.active_chunk_id = None
            self.video_samples = []
            self.next_video_sample_s = 1.0 / options.video_fps
            self.frame_sink = None
            self.hold_phase = "inference_hold"
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
            recording = self._recording_kwargs(kwargs, self.hold_phase)
            if self.fixture_timing is not None:
                advanced = self.fixture_timing.advance(duration, **recording)
                if callback_convention == "source_run":
                    self._source_action_callback_offset_s += float(advanced)
            else:
                advanced = super()._apply_sync_latency(env, duration, **recording)
            self.rollout_sim_time_s += float(advanced)
            self._motion_event("hold_completed", applied_latency_s=float(advanced))
            return advanced

        def _recording_kwargs(self, kwargs, phase):
            callback = kwargs.get("on_substep")
            if not options.save_videos or (
                callback is None and self.frame_sink is None
            ):
                return kwargs
            base_sim_time_s = self.rollout_sim_time_s

            def record(advanced_s):
                absolute_sim_time_s = base_sim_time_s + float(advanced_s)
                if callback is not None:
                    callback(advanced_s)
                # Source callback samples at this threshold after a real physics
                # substep. Verify the sample count against its final manifest.
                if absolute_sim_time_s + 1e-9 >= self.next_video_sample_s:
                    sample = {
                        "frame_index": len(self.video_samples),
                        "media_time_s": len(self.video_samples) / options.video_fps,
                        "nominal_sim_time_s": self.next_video_sample_s,
                        "sample_sim_time_s": absolute_sim_time_s,
                        "wall_completed_s": time.monotonic(),
                        "phase": phase,
                        "action_step": self.action_step,
                        "active_chunk_id": self.active_chunk_id,
                    }
                    # Record physical sampling before either encoder writes. A
                    # failed second stream must not erase the first stream's map.
                    self.video_samples.append(sample)
                    self.next_video_sample_s += 1.0 / options.video_fps
                    if self.frame_sink is not None:
                        sample["sample_status"] = "started"
                        try:
                            self.frame_sink(self.env, sample)
                            sample["sample_status"] = "complete"
                        except BaseException as error:
                            sample.update(
                                sample_status="error",
                                error=f"{type(error).__name__}: {error}",
                            )
                            raise
                        finally:
                            sample["wall_completed_s"] = time.monotonic()

            return {**kwargs, "on_substep": record}

    return PhysicalEpisode
