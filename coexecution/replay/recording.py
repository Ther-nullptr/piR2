"""Bounded owned robot observations; all codecs and file I/O run after control."""

from pathlib import Path

from .schema import VERSION, interval, optional_number, write_json


def robot_state(observation):
    """Pinned LIBERO roll/pitch/yaw fields contain axis-angle, not Euler angles."""

    def vector(keys):
        values = []
        for key in keys:
            value = observation.get("state." + key)
            if value is None:
                return None
            flat = (
                value.reshape(-1).tolist() if hasattr(value, "reshape") else list(value)
            )
            values.extend(float(optional_number(float(item))) for item in flat)
        return values

    return {
        "position_xyz_m": vector(("x", "y", "z")),
        "rotation_rotvec_rad": vector(("roll", "pitch", "yaw")),
        "gripper_qpos": vector(("gripper",)),
    }


class ReplayRecorder:
    """Stores one initial and bounded post-step frames, with explicit wall mapping.

    Recording adds owned array copies to the debug control run, never encoding,
    storage or neural computation. Overflow stops recording without stopping the
    physical controller; partial evidence is explicitly labelled.
    """

    def __init__(self, *, max_frames=1024, max_bytes=512 * 1024 * 1024):
        if (
            type(max_frames) is not int
            or max_frames < 1
            or type(max_bytes) is not int
            or max_bytes < 1
        ):
            raise ValueError("Replay frame and byte limits must be positive integers")
        self.max_frames, self.max_bytes = max_frames, max_bytes
        self.frames, self.mapping = {}, []
        self.used_bytes = self.dropped_frames = 0
        self.error = None
        self.initial_state = None
        self._finished = False

    def _capture(self, observation, *, tick, phase, wall_s, computed_s):
        import numpy as np

        if self._finished:
            raise RuntimeError("Recorder already finished")
        if self.error is not None:
            self.dropped_frames += 1
            return False
        optional_number(wall_s)
        optional_number(computed_s)
        if self.mapping and wall_s < self.mapping[-1]["wall_s"]:
            raise ValueError("Frame timestamps must be monotonic")
        cameras = {
            key[6:]: np.asarray(value)
            for key, value in observation.items()
            if key.startswith("video.")
        }
        if not cameras:
            self.error = "No real camera observations were present"
            return False
        if self.frames and set(cameras) != set(self.frames):
            self.error = "Camera set changed during recording"
            return False
        if any(
            value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3
            for value in cameras.values()
        ):
            self.error = "Expected uint8 HWC RGB camera observations"
            return False
        size = sum(value.nbytes for value in cameras.values())
        if (
            len(self.mapping) >= self.max_frames
            or self.used_bytes + size > self.max_bytes
            or self.dropped_frames
        ):
            self.dropped_frames += 1
            return False
        try:
            # Stage the entire camera group and its metadata before publishing.
            # The lists copy references only; existing pixel buffers stay owned.
            owned = {
                camera: np.array(value, copy=True, order="C")
                for camera, value in cameras.items()
            }
            frames = {
                camera: [*self.frames.get(camera, []), value]
                for camera, value in owned.items()
            }
            mapping = [
                *self.mapping,
                {
                    "frame_id": f"frame-{len(self.mapping)}",
                    "frame_index": len(self.mapping),
                    "tick": tick,
                    "phase": phase,
                    "wall_s": wall_s,
                    "observation_computed_s": computed_s,
                    "media_time_s": len(self.mapping) / 20.0,
                },
            ]
            used_bytes = self.used_bytes + size
        except MemoryError:
            self.error = "Frame recording failed: MemoryError"
            self.dropped_frames += 1
            return False
        self.frames, self.mapping, self.used_bytes = frames, mapping, used_bytes
        return True

    def initial(self, observation, *, capture_s, computed_s=None):
        if self.mapping:
            raise RuntimeError("Initial replay observation was already recorded")
        self.initial_state = robot_state(observation)
        self._capture(
            observation,
            tick=None,
            phase="initial",
            wall_s=capture_s,
            computed_s=computed_s,
        )

    def step(self, before, after, tick):
        interval(tick["applied_s"], tick["next_observation_computed_s"])
        if not self._capture(
            after,
            tick=tick["tick"],
            phase="post_step",
            wall_s=tick["next_observation_computed_s"],
            computed_s=tick["next_observation_computed_s"],
        ):
            return None
        return {
            "before": robot_state(before),
            "after": robot_state(after),
            "before_capture_s": tick["sensor_capture_s"],
            "before_release_s": tick["sensor_release_s"],
            "after_computed_s": tick["next_observation_computed_s"],
            "after_frame_id": self.mapping[-1]["frame_id"],
        }

    def status(self):
        return {
            "status": "failed"
            if self.error
            else "overflow"
            if self.dropped_frames
            else "recorded",
            "error": self.error,
            "recorded_frames": len(self.mapping),
            "dropped_frames": self.dropped_frames,
            "max_frames": self.max_frames,
            "max_bytes": self.max_bytes,
            "frame_bytes": self.used_bytes,
            "rotation_representation": "axis_angle_rotation_vector_rad",
            "command_units": "normalized_LIBERO_controller_commands",
            "instrumented": True,
            "scope": "Owned debug copies during control; encoding and disk writes after control",
        }

    def finish(self, output, stem, *, encoder=None):
        if self._finished:
            raise RuntimeError("Recorder already finished")
        self._finished = True
        output = Path(output)
        if Path(stem).name != stem or stem in ("", ".", ".."):
            raise ValueError("Replay stem must be a simple name")
        output.mkdir(parents=True, exist_ok=True)
        if encoder is None:
            from scripts.libero_observations import write_video

            encoder = write_video
        media = {}
        for camera, frames in self.frames.items():
            if Path(camera).name != camera or camera in (".", ".."):
                self.error = "Invalid camera identifier"
                continue
            path = output / f"{stem}-{camera}.webm"
            try:
                encoder(path, frames, codec="libvpx-vp9", fps=20)
                media[camera] = {
                    "path": path.name,
                    "mime": "video/webm",
                    "frame_count": len(frames),
                    "fps": 20,
                }
            except (OSError, ValueError, RuntimeError) as error:
                self.error = f"Video encoding failed: {type(error).__name__}: {error}"
                media[camera] = {"status": "failed", "error": self.error}
        data = {
            "schema": "pir2.robot-recording",
            "version": VERSION,
            **self.status(),
            "initial_state": self.initial_state,
            "frames": self.mapping,
            "media": media,
        }
        path = output / f"{stem}-replay.json"
        write_json(path, data)
        return {"path": path.name, **self.status()}
