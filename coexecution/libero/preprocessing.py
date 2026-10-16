"""Explicit image-information reduction, outside numerical optimization scopes.

Imports and configuration validation do not load model or image dependencies.
Apply before collation/preparation and close after all inference/Graph scopes.
Changing resolution inside a live inference session is intentionally unsupported.
"""

import threading
from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps


@dataclass(frozen=True)
class ImageResolutionConfig:
    size: int | None = None

    def __post_init__(self):
        if self.size is not None and (type(self.size) is not int or self.size < 1):
            raise ValueError("Image resolution must be a positive integer or None")


def add_image_arguments(parser):
    parser.add_argument(
        "--image-resolution",
        type=int,
        help="Fixed square inference size after the checkpoint crop; omit to preserve checkpoint preprocessing",
    )


def image_config(args):
    return ImageResolutionConfig(getattr(args, "image_resolution", None))


def _image_grids(value):
    grids = []
    if isinstance(value, Mapping):
        for name, item in value.items():
            if name == "image_grid_thw":
                if getattr(item, "is_cuda", False):
                    raise ValueError("Resolution audit requires CPU collation metadata")
                grids.extend(item.tolist() if hasattr(item, "tolist") else item)
            else:
                grids.extend(_image_grids(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            grids.extend(_image_grids(item))
    return grids


class ImageResolutionScope:
    """One startup-fixed resolution for every processor consumer of a policy.

    Keep the checkpoint's complete deterministic eval transform, then append an
    area resize. Training transforms and checkpoint metadata remain untouched.
    A disabled scope is a strict no-op, including on unsupported processors.
    The caller owns serial preparation and quiescent entry/exit, just as for
    GrootOptimizations. Nested ownership of the same processor is rejected.
    """

    def __init__(self, policy, config=None):
        self.policy = policy
        self.config = ImageResolutionConfig() if config is None else config
        self.stack = ExitStack()
        self.lock = threading.Lock()
        self.active = False
        self.epoch = 0
        self.observations = 0
        self.grids = []
        self.feature_shapes = {}
        self.contract = {}

    def _replace(self, owner, name, value):
        existed, original = name in vars(owner), getattr(owner, name, None)
        self.stack.callback(
            setattr if existed else delattr,
            owner,
            name,
            *([original] if existed else []),
        )
        setattr(owner, name, value)

    def __enter__(self):
        if self.active:
            raise RuntimeError("Image resolution scope is already active")
        self.active = True
        self.epoch += 1
        if self.config.size is None:
            return self
        try:
            self._install()
        except BaseException:
            self.close()
            raise
        return self

    def _install(self):
        processor = self.policy.processor
        if getattr(processor, "_pir2_image_resolution", None) is not None:
            raise RuntimeError("Image processor already has a resolution scope")
        if getattr(processor, "training", False) or self.policy.model.training:
            raise ValueError("Image resolution overrides require eval mode")
        if not processor.use_albumentations:
            raise ValueError(
                "Image resolution requires checkpoint albumentations preprocessing"
            )
        native = processor.image_target_size
        size = self.config.size
        if (
            not isinstance(native, (tuple, list))
            or len(native) != 2
            or native[0] != native[1]
            or size > native[0]
        ):
            raise ValueError(
                "Image resolution requires a square checkpoint target and cannot upscale it"
            )
        collators = [processor.collator, self.policy.collate_fn]
        if getattr(self.policy.model, "collator", None) is not None:
            collators.append(self.policy.model.collator)
        consumers = [processor.processor, *(item.processor for item in collators)]
        pixels = {id(item.image_processor): item.image_processor for item in consumers}
        geometries = {(item.patch_size, item.merge_size) for item in pixels.values()}
        if len(geometries) != 1:
            raise ValueError("Qwen image consumers have inconsistent patch geometry")
        patch, merge = next(iter(geometries))
        if any(
            type(value) is not int or value < 1 for value in (patch, merge)
        ) or size % (patch * merge):
            raise ValueError(
                "Image resolution must be divisible by Qwen patch_size * merge_size"
            )
        for item in pixels.values():
            if (
                not isinstance(item.size, Mapping)
                or not {"shortest_edge", "longest_edge"} <= item.size.keys()
            ):
                raise ValueError("Unsupported Qwen image size configuration")
        import albumentations as A
        import cv2

        transform = processor.eval_image_transform
        if not isinstance(transform, A.Compose):
            raise TypeError("Expected the checkpoint's albumentations Compose")
        resized = transform
        if size != native[0]:
            # Preserve Compose options/custom mask transforms and do not reseed
            # or mutate transform objects still owned by the checkpoint scope.
            resized = deepcopy(transform)
            resize = A.Resize(size, size, interpolation=cv2.INTER_AREA)
            if getattr(resized, "additional_targets", None):
                resize.add_targets(resized.additional_targets)
            resized.transforms.append(resize)
        self.contract = {
            "checkpoint_target": list(native),
            "output_size": [size, size],
            "crop_policy": "checkpoint eval transform followed by INTER_AREA resize",
            "patch_size": patch,
            "merge_size": merge,
            "pixel_consumers": len(pixels),
        }
        self._replace(processor, "_pir2_image_resolution", self)
        self._replace(processor, "eval_image_transform", resized)
        for item in pixels.values():
            self._replace(
                item,
                "size",
                {**item.size, "shortest_edge": size**2, "longest_edge": size**2},
            )
            for name in ("min_pixels", "max_pixels"):
                if hasattr(item, name):
                    self._replace(item, name, size**2)
        original = self.policy._collate_observation
        epoch = self.epoch

        @wraps(original)
        def collate(*args, **kwargs):
            if not self.lock.acquire(blocking=False):
                raise RuntimeError("Image resolution requires serial collation")
            try:
                if not self.active or epoch != self.epoch:
                    raise RuntimeError("Image resolution scope is closed")
                result = original(*args, **kwargs)
                grids = _image_grids(result)
                expected = [1, size // patch, size // patch]
                if not grids or any(list(grid) != expected for grid in grids):
                    raise RuntimeError(
                        "Requested image resolution did not reach the actual Qwen patch grid"
                    )
                self.grids = [list(grid) for grid in grids]
                self.observations += 1
                return result
            finally:
                self.lock.release()

        self._replace(self.policy, "_collate_observation", collate)

    def record_features(self, features):
        self.feature_shapes = {
            name: list(value.shape)
            for name, value in features.items()
            if hasattr(value, "shape")
        }

    def configuration(self):
        return {"requested_size": self.config.size, **self.contract}

    def evidence(self):
        return {
            **self.configuration(),
            "observations": self.observations,
            "actual_grid_thw": [list(grid) for grid in self.grids],
            "visual_tokens_per_image": [
                grid[0] * grid[1] * grid[2] // self.contract["merge_size"] ** 2
                for grid in self.grids
            ],
            "feature_shapes": {
                name: list(shape) for name, shape in self.feature_shapes.items()
            },
        }

    def close(self):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Cannot close image resolution during active collation")
        try:
            self.stack.close()
            self.active = False
        finally:
            self.lock.release()

    def __exit__(self, *_exc):
        self.close()
