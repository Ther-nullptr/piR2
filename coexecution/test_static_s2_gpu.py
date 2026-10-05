"""GPU-only exactness and invalidation checks for S2 static metadata.

Run with exactly one visible, clean GPU. Collection does not create a CUDA
context; the module fixture loads the real checkpoint only during execution.
"""

from pathlib import Path

import pytest
import torch

from coexecution.fusion import FusionPatch
from coexecution.static_s2 import StaticS2
from coexecution.workload import PiR2Workload


@pytest.fixture(scope="module")
def workload():
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(__file__).resolve().parents[1]
    return PiR2Workload(root / "outputs/pir2-so100-smoke/checkpoint-10")


def snapshot(values):
    return {key: value.detach().clone() for key, value in values.items()}


def exact(reference, candidate):
    assert reference.keys() == candidate.keys()
    for key in reference:
        torch.testing.assert_close(reference[key], candidate[key], rtol=0, atol=0)
        assert reference[key].stride() == candidate[key].stride()
        if candidate[key].is_floating_point():
            assert torch.isfinite(candidate[key]).all()


@torch.inference_mode()
def test_all_real_frames_and_outer_rope_fusion_are_exact(workload):
    references = [snapshot(workload.slow(i)) for i in range(8)]
    with FusionPatch(workload, "rope") as fusion, StaticS2(workload) as patch:
        for index, reference in enumerate(references):
            exact(reference, workload.slow(index))
        evidence = patch.evidence()
        assert evidence["calls"] == 8
        assert evidence["cache_hits"] == 8
        assert evidence["invalidations"] == 0
        assert evidence["fallbacks"] == 0
        assert evidence["vision_forwards"] == 8
        assert fusion.evidence()["calls"]["vision_rope/fused"] > 0


@torch.inference_mode()
def test_changed_pixels_are_recomputed_without_metadata_rebuild(workload):
    eager = workload.slow
    with StaticS2(workload) as patch:
        before = snapshot(workload.slow(0))
        preparations = patch.evidence()["preparations"]
        workload.backbone_inputs[0]["pixel_values"] = (
            workload.backbone_inputs[0]["pixel_values"].clone() * 0.5
        )
        reference = snapshot(eager(0))
        actual = workload.slow(0)
        exact(reference, actual)
        assert not torch.equal(before["backbone_features"], actual["backbone_features"])
        assert patch.evidence()["preparations"] == preparations
        assert patch.evidence()["invalidations"] == 0
        assert patch.evidence()["vision_forwards"] == 2


@pytest.mark.parametrize("key", ["input_ids", "attention_mask", "image_grid_thw"])
@torch.inference_mode()
def test_same_shape_metadata_content_change_invalidates(workload, key):
    eager = workload.slow
    with StaticS2(workload) as patch:
        workload.slow(0)
        if key == "input_ids":
            workload.backbone_inputs[0][key][0, -1] = 42
        elif key == "attention_mask":
            workload.backbone_inputs[0][key][0, -1] = 0
        else:
            grid = workload.backbone_inputs[0][key]
            grid[:, 1] //= 2
            grid[:, 2] *= 2
        reference = snapshot(eager(0))
        exact(reference, workload.slow(0))
        assert patch.evidence()["invalidations"] == 1
        assert patch.evidence()["preparations"] >= 2


@torch.inference_mode()
def test_replacement_and_token_length_change_rebuild_safely(workload):
    eager = workload.slow
    with StaticS2(workload) as patch:
        workload.slow(0)
        data = workload.backbone_inputs[0]
        # Replacement under inference_mode creates an inference tensor. The
        # adapter must convert it to a version-tracked tensor before reuse.
        data["input_ids"] = data["input_ids"].clone()
        exact(snapshot(eager(0)), workload.slow(0))
        assert not torch.is_inference(data["input_ids"])
        assert patch.evidence()["invalidations"] == 1
        data["input_ids"] = torch.cat(
            (data["input_ids"], data["input_ids"][:, -1:]), dim=1
        )
        data["attention_mask"] = torch.cat(
            (data["attention_mask"], data["attention_mask"][:, -1:]), dim=1
        )
        exact(snapshot(eager(0)), workload.slow(0))
        assert patch.evidence()["invalidations"] == 2


@torch.inference_mode()
def test_prepared_metadata_works_on_another_cuda_stream(workload):
    reference = snapshot(workload.slow(0))
    with StaticS2(workload):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = snapshot(workload.slow(0))
        torch.cuda.current_stream().wait_stream(stream)
        exact(reference, actual)


@torch.inference_mode()
def test_context_restores_inputs_methods_and_rope_state_on_error(workload):
    inputs = workload.backbone_inputs
    slow = workload.slow
    inner = workload.model.backbone.model.model
    methods = {
        "vision": inner.visual.forward,
        "attention": inner.visual.blocks[0].attn.forward,
        "features": inner.get_image_features,
        "placeholder": inner.get_placeholder_mask,
    }
    rope_deltas = inner.rope_deltas
    with pytest.raises(RuntimeError, match="intentional test"), StaticS2(workload):
        assert workload.backbone_inputs is not inputs
        workload.slow(0)
        raise RuntimeError("intentional test")
    assert workload.backbone_inputs is inputs
    assert workload.slow == slow
    assert inner.visual.forward == methods["vision"]
    assert inner.visual.blocks[0].attn.forward == methods["attention"]
    assert inner.get_image_features == methods["features"]
    assert inner.get_placeholder_mask == methods["placeholder"]
    assert inner.rope_deltas is rope_deltas


@torch.inference_mode()
def test_unsupported_video_and_bad_placeholder_fail_explicitly(workload):
    with StaticS2(workload):
        data = workload.backbone_inputs[0]
        data["pixel_values_videos"] = data["pixel_values"]
        with pytest.raises(ValueError, match="image-only"):
            workload.slow(0)
        del data["pixel_values_videos"]
        image_id = workload.model.backbone.model.config.image_token_id
        data["input_ids"][data["input_ids"] == image_id] = 42
        with pytest.raises(ValueError, match="Image features and image tokens"):
            workload.slow(0)
