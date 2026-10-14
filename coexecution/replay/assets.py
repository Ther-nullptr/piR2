"""Optional raw observation image/state assets, exported outside model timing."""

from pathlib import Path

from .schema import write_json


def export_observations(output, observations, identities, *, image_writer=None):
    import numpy as np

    if len(observations) != len(identities):
        raise ValueError("One identity is required per observation")
    if image_writer is None:
        from PIL import Image

        def image_writer(path, value):
            Image.fromarray(value).save(path)

    output = Path(output)
    directory = output / "replay-assets"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (observation, identity) in enumerate(zip(observations, identities)):
        assets = []
        for camera_index, (camera, value) in enumerate(
            sorted(observation["video"].items())
        ):
            array = np.asarray(value)
            if array.dtype != np.uint8 or array.ndim < 3 or array.shape[-1] != 3:
                raise ValueError("Raw replay assets require uint8 HWC RGB observations")
            for frame_index, frame in enumerate(array.reshape((-1, *array.shape[-3:]))):
                path = (
                    directory
                    / f"observation-{index}-camera-{camera_index}-frame-{frame_index}.png"
                )
                image_writer(path, frame)
                assets.append(
                    {
                        "camera": camera,
                        "frame_index": frame_index,
                        "path": str(path.relative_to(output)),
                        "mime": "image/png",
                    }
                )
        records.append(
            {
                **identity,
                "assets": assets,
                "state": {
                    key: np.asarray(value).tolist()
                    for key, value in observation["state"].items()
                },
            }
        )
    write_json(output / "observations-replay.json", records)
    return records
