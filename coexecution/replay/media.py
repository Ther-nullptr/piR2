"""Explicit media manifest. No caller-supplied path becomes an HTTP route."""

import json
import math
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path

from .schema import digest, source, under_root


def _video_info(path):
    """Read presentation timestamps from decoded frames, not nominal frame rate."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-threads",
            "2",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,time_base:frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    if len(streams) != 1:
        raise ValueError("Replay video needs one selected video stream")
    times = [
        float(frame["best_effort_timestamp_time"]) for frame in info.get("frames", [])
    ]
    if (
        not times
        or any(not math.isfinite(t) or t < 0 for t in times)
        or any(right <= left for left, right in pairwise(times))
    ):
        raise ValueError(
            "Replay video needs strictly increasing presentation timestamps"
        )
    return {**streams[0], "timestamps_s": times}


class MediaRegistry:
    def __init__(self, output):
        self.output, self.assets = Path(output), {}

    def add(self, root, relative, *, mime):
        suffixes = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "video/webm": ".webm",
            "video/mp4": ".mp4",
        }
        if mime not in suffixes:
            raise ValueError("Unsupported replay media type")
        path = under_root(root, relative)
        provenance = source(path)
        identity = "media-" + digest(provenance)[:24]
        target = Path("media") / (identity + suffixes[mime])
        if identity not in self.assets:
            destination = self.output / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            self.assets[identity] = {
                "id": identity,
                "path": str(target),
                "mime": mime,
                "source": provenance,
            }
        return identity

    def browser_video(self, root, relative, *, source_media_times):
        """Create a lossless VP9 derivative, preserving decoded frame order.

        WebM timestamps may round to milliseconds. Return their actual values so
        callers retain their simulation mapping while seeking the right frame.
        The original file is never modified or substituted in its provenance.
        """
        path = under_root(root, relative)
        provenance = source(path)
        original = _video_info(path)
        if len(source_media_times) != len(original["timestamps_s"]):
            raise ValueError("Replay source frame count differs from recorded mapping")
        if any(
            not math.isfinite(expected) or abs(expected - actual) > 1e-6
            for expected, actual in zip(source_media_times, original["timestamps_s"])
        ):
            raise ValueError("Replay source timestamp differs from recorded mapping")
        options = [
            "-map",
            "0:v:0",
            "-an",
            "-vsync",
            "0",
            "-c:v",
            "libvpx-vp9",
            "-lossless",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-row-mt",
            "1",
            "-threads",
            "2",
            "-deadline",
            "good",
            "-cpu-used",
            "4",
            "-g",
            "60",
        ]
        version = subprocess.run(
            ["ffmpeg", "-version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0]
        identity = "media-" + digest([provenance, options, version])[:24]
        target = Path("media") / (identity + ".webm")
        if identity not in self.assets:
            destination = self.output / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-nostdin",
                        "-n",
                        "-threads",
                        "2",
                        "-i",
                        str(path),
                        *options,
                        str(destination),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                converted = _video_info(destination)
                if converted["codec_name"] != "vp9" or any(
                    original[key] != converted[key] for key in ("width", "height")
                ):
                    raise ValueError("Browser derivative codec or dimensions differ")
                if len(original["timestamps_s"]) != len(converted["timestamps_s"]):
                    raise ValueError("Browser derivative changed video frame count")
                if any(
                    abs(before - after) > 0.001001
                    for before, after in zip(
                        original["timestamps_s"], converted["timestamps_s"]
                    )
                ):
                    raise ValueError(
                        "Browser derivative changed presentation timestamp order"
                    )
            except Exception:
                destination.unlink(missing_ok=True)
                raise
            self.assets[identity] = {
                "id": identity,
                "path": str(target),
                "mime": "video/webm",
                "source": provenance,
                "derivation": {
                    "codec": "vp9",
                    "lossless": True,
                    "frame_order": "passthrough",
                    "frame_count": len(converted["timestamps_s"]),
                    "source_timestamps_s": original["timestamps_s"],
                    "media_timestamps_s": converted["timestamps_s"],
                    "ffmpeg_version": version,
                    "encoding_arguments": options,
                    "output": source(destination),
                },
            }
        asset = self.assets[identity]
        return {
            "media_id": identity,
            "mime": asset["mime"],
            "media_times_s": asset["derivation"]["media_timestamps_s"],
        }
