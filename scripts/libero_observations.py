"""Shared observation packing and video output, without an obsolete evaluator."""

import av
import numpy as np


def batch_observation(observation):
    return {
        key: [value]
        if isinstance(value, str)
        else np.asarray(
            value, dtype=np.uint8 if key.startswith("video.") else np.float32
        )[None, None]
        for key, value in observation.items()
    }


def write_video(path, frames):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=20)
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(frame, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
