"""Resolution ownership, real-grid admission and entrypoint wiring; no models."""

import argparse
import json
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from coexecution.libero.preprocessing import (
    ImageResolutionConfig,
    ImageResolutionScope,
    add_image_arguments,
    image_config,
)


@pytest.fixture
def policy(monkeypatch):
    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms

    class Resize:
        def __init__(self, height, width, interpolation):
            self.height, self.width, self.interpolation = height, width, interpolation

    monkeypatch.setitem(
        sys.modules, "albumentations", SimpleNamespace(Compose=Compose, Resize=Resize)
    )
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(INTER_AREA=3))

    def consumer():
        return SimpleNamespace(
            image_processor=SimpleNamespace(
                size={"shortest_edge": 65536, "longest_edge": 1048576},
                min_pixels=65536,
                max_pixels=1048576,
                patch_size=16,
                merge_size=2,
            )
        )

    template, collator, model_collator = consumer(), consumer(), consumer()
    processor = SimpleNamespace(
        training=False,
        use_albumentations=True,
        image_target_size=[256, 256],
        image_crop_size=[230, 230],
        eval_image_transform=Compose(["trained-crop", "trained-resize"]),
        train_image_transform=object(),
        processor=template,
        collator=SimpleNamespace(processor=collator),
    )

    class Policy:
        grid_override = None
        callback = None

        def _collate_observation(self, observation):
            if self.callback is not None:
                self.callback()
            side = (
                int(
                    self.collate_fn.processor.image_processor.size["shortest_edge"]
                    ** 0.5
                )
                // 16
            )
            grid = self.grid_override or [1, side, side]
            return {"inputs": {"image_grid_thw": [grid, grid]}}, observation

    value = Policy()
    value.processor, value.collate_fn = processor, processor.collator
    value.model = SimpleNamespace(
        training=False, collator=SimpleNamespace(processor=model_collator)
    )
    value.pixels = [
        item.image_processor for item in (template, collator, model_collator)
    ]
    return value


def test_disabled_scope_is_strictly_noop_without_runtime_imports():
    program = """
import sys
class RejectRuntime:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'gr00t', 'cv2', 'albumentations'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectRuntime())
from coexecution.libero.preprocessing import ImageResolutionScope
policy = object()
with ImageResolutionScope(policy) as scope:
    assert scope.evidence()['requested_size'] is None
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("size", [False, True, 0, -1, 128.0, "128", [], {}])
def test_invalid_configuration_rejected_without_loading_dependencies(size):
    with pytest.raises(ValueError, match="positive integer"):
        ImageResolutionConfig(size)


@pytest.mark.parametrize("size", [128, 192, 256])
def test_all_qwen_consumers_shrink_and_original_objects_are_restored(policy, size):
    processor = policy.processor
    original = processor.eval_image_transform
    training = processor.train_image_transform
    sizes = [item.size for item in policy.pixels]
    scope = ImageResolutionScope(policy, ImageResolutionConfig(size))
    for _ in range(2):
        with scope:
            for item in policy.pixels:
                assert item.size == {"shortest_edge": size**2, "longest_edge": size**2}
                assert item.min_pixels == item.max_pixels == size**2
            assert processor.eval_image_transform.transforms[:2] == original.transforms
            assert len(processor.eval_image_transform.transforms) == (
                2 if size == 256 else 3
            )
            assert processor.train_image_transform is training
            policy._collate_observation(object())
            evidence = scope.evidence()
            assert evidence["actual_grid_thw"] == [[1, size // 16, size // 16]] * 2
            assert evidence["visual_tokens_per_image"] == [(size // 32) ** 2] * 2
            evidence["actual_grid_thw"][0][1] = 99
            assert scope.evidence()["actual_grid_thw"][0][1] == size // 16
        assert processor.eval_image_transform is original
        assert [item.size for item in policy.pixels] == sizes
        assert all(
            item.size is saved for item, saved in zip(policy.pixels, sizes, strict=True)
        )
        assert all(
            item.min_pixels == 65536 and item.max_pixels == 1048576
            for item in policy.pixels
        )
        assert "_collate_observation" not in vars(policy)
        assert "_pir2_image_resolution" not in vars(processor)


def test_failed_grid_audit_restores_processor_and_reports_no_successful_observation(
    policy,
):
    original = policy.processor.eval_image_transform
    policy.grid_override = [1, 16, 16]
    with (
        pytest.raises(RuntimeError, match="actual Qwen patch grid"),
        ImageResolutionScope(policy, ImageResolutionConfig(128)) as scope,
    ):
        policy._collate_observation(None)
    assert scope.observations == 0
    assert policy.processor.eval_image_transform is original
    assert all(item.size["shortest_edge"] == 65536 for item in policy.pixels)


@pytest.mark.parametrize("size", [127, 144, 288])
def test_illegal_grid_or_upscale_is_rejected_atomically(policy, size):
    original = policy.processor.eval_image_transform
    with (
        pytest.raises(ValueError),
        ImageResolutionScope(policy, ImageResolutionConfig(size)),
    ):
        pytest.fail("Invalid resolution admitted")
    assert policy.processor.eval_image_transform is original
    assert all(item.size["shortest_edge"] == 65536 for item in policy.pixels)


def test_nested_scope_does_not_stack_resizes_or_restore_another_owners_state(policy):
    with ImageResolutionScope(policy, ImageResolutionConfig(128)) as first:
        with (
            pytest.raises(RuntimeError, match="already has"),
            ImageResolutionScope(policy, ImageResolutionConfig(192)),
        ):
            pytest.fail("Nested scope admitted")
        assert policy.pixels[0].size["shortest_edge"] == 128**2
        with pytest.raises(RuntimeError, match="already active"):
            first.__enter__()
    with ImageResolutionScope(policy, ImageResolutionConfig(256)):
        assert len(policy.processor.eval_image_transform.transforms) == 2
        first.close()
        assert policy.pixels[0].size["shortest_edge"] == 256**2


def test_active_collation_rejects_close_and_stale_collation_rejects_reuse(policy):
    scope = ImageResolutionScope(policy, ImageResolutionConfig(128))
    with scope:
        policy.callback = scope.close
        with pytest.raises(RuntimeError, match="active collation"):
            policy._collate_observation(None)
        policy.callback = None
        retained = policy._collate_observation
        assert retained(None)
    with pytest.raises(RuntimeError, match="closed"):
        retained(None)
    with scope, pytest.raises(RuntimeError, match="closed"):
        retained(None)


def test_aliased_consumers_are_mutated_and_restored_once(policy):
    policy.model.collator.processor = policy.collate_fn.processor
    policy.processor.processor = policy.collate_fn.processor
    pixel = policy.collate_fn.processor.image_processor
    original = pixel.size
    with ImageResolutionScope(policy, ImageResolutionConfig(192)) as scope:
        assert scope.evidence()["pixel_consumers"] == 1
    assert pixel.size is original


def test_feature_shapes_are_metadata_only_and_evidence_is_json_serializable(policy):
    with ImageResolutionScope(policy, ImageResolutionConfig(128)) as scope:
        policy._collate_observation(None)
        scope.record_features(
            {"backbone_features": SimpleNamespace(shape=(1, 57, 2048))}
        )
        assert json.loads(json.dumps(scope.evidence()))["feature_shapes"] == {
            "backbone_features": [1, 57, 2048]
        }


def test_cli_defaults_and_explicit_resolution_are_independent_of_optimization():
    parser = argparse.ArgumentParser()
    add_image_arguments(parser)
    assert image_config(parser.parse_args([])).size is None
    assert image_config(parser.parse_args(["--image-resolution", "192"])).size == 192


@pytest.mark.parametrize(
    "module, arguments",
    [
        ("coexecution.libero.service", []),
        ("coexecution.libero.measurement", []),
        ("coexecution.profiling.worker", []),
        ("coexecution.profiling.flow", []),
        ("coexecution.profiling.campaign", ["prepare"]),
    ],
)
def test_existing_entrypoints_expose_resolution_without_loading_runtime(
    module, arguments
):
    program = """
import sys, runpy
class RejectRuntime:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'gr00t', 'cv2', 'albumentations'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectRuntime())
name = sys.argv[1]
sys.argv = [name, *sys.argv[2:], '--help']
runpy.run_module(name, run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", program, module, *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--image-resolution" in result.stdout


def test_runtime_observations_do_not_change_resume_identity(policy):
    with ImageResolutionScope(policy, ImageResolutionConfig(128)) as scope:
        before = scope.configuration()
        policy._collate_observation(None)
        scope.record_features(
            {"backbone_features": SimpleNamespace(shape=(1, 57, 2048))}
        )
        assert scope.configuration() == before
        assert scope.evidence()["observations"] == 1


@pytest.mark.parametrize("failure", [False, True])
def test_service_installs_before_optimization_and_restores_after_graph_scope(
    policy, monkeypatch, failure
):
    from coexecution.libero import service

    order = []

    @contextmanager
    def optimized(core, config):
        assert core.collate_fn.processor.image_processor.size["shortest_edge"] == 128**2
        order.append("graphs_enter")
        if failure:
            raise RuntimeError("Capture setup failed")
        try:
            yield object()
        finally:
            assert (
                core.collate_fn.processor.image_processor.size["shortest_edge"]
                == 128**2
            )
            order.append("graphs_close")

    monkeypatch.setattr(service, "GrootOptimizations", optimized)
    backend = SimpleNamespace(core=policy, stream=object())
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            device=lambda _: nullcontext(), stream=lambda _: nullcontext()
        )
    )
    context = service.worker_scope(
        backend, {"device": "cuda:0", "image_resolution": 128}, object(), torch
    )
    if failure:
        with pytest.raises(RuntimeError, match="Capture setup"), context:
            pytest.fail("Failed optimization entered")
    else:
        with context:
            assert backend.image_resolution.config.size == 128
            policy._collate_observation(None)
        assert order == ["graphs_enter", "graphs_close"]
    assert policy.pixels[0].size["shortest_edge"] == 65536
    assert not hasattr(backend, "image_resolution")
