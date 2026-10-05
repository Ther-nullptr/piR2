"""The actual pretrained GR00T backbone and piR2 head on one CUDA device."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.decoupled_policy import DecoupledGr00tPolicy
from transformers.feature_extraction_utils import BatchFeature

ROOT = Path(__file__).resolve().parents[1]


class PiR2Workload:
    def __init__(self, checkpoint, samples=8, features_only=True):
        assert torch.cuda.device_count() == 1, "Expose exactly one physical GPU"
        self.policy = DecoupledGr00tPolicy(
            EmbodimentTag.NEW_EMBODIMENT, str(checkpoint), device="cuda:0"
        )
        self.model = self.policy.model.eval()
        self.features_only = features_only
        self._pre_final_norm = None

        def capture_pre_final_norm(_module, inputs):
            self._pre_final_norm = inputs[0]

        # Only the S2 worker calls this backbone; S1 reads cache copies.
        self._norm_hook = self.model.backbone.model.model.language_model.norm.register_forward_pre_hook(
            capture_pre_final_norm
        )
        self.backbone_inputs, self.action_inputs = [], []
        loader = LeRobotEpisodeLoader(
            ROOT / "upstream/learning/Isaac-GR00T/demo_data/cube_to_bowl_5",
            self.policy.modality_configs,
        )
        trajectory = loader[0]
        modalities = deepcopy(self.policy.modality_configs)
        modalities.pop("action", None)
        for i in range(samples):
            data = extract_step_data(
                trajectory, i * 3, modalities, EmbodimentTag.NEW_EMBODIMENT
            )
            observation = {
                "state": {k: np.asarray(v)[None] for k, v in data.states.items()},
                "video": {k: np.asarray(v)[None] for k, v in data.images.items()},
                "language": {
                    k: [[data.text]] for k in modalities["language"].modality_keys
                },
            }
            collated, _ = self.policy._collate_observation(observation)
            backbone, action = self.model.prepare_input(collated["inputs"])
            self.backbone_inputs.append(backbone)
            self.action_inputs.append(
                {k: action[k] for k in ["state", "embodiment_id"]}
            )
        self.reference_checks = []
        with torch.inference_mode():
            for i in range(min(samples, 2)):
                reference = self.model.backbone(self.backbone_inputs[i])
                optimized = self.slow(i)
                differences = {}
                for key in reference:
                    if reference[key].is_floating_point():
                        difference = (
                            reference[key].float() - optimized[key].float()
                        ).abs()
                        differences[key] = float(difference.max())
                        torch.testing.assert_close(
                            reference[key], optimized[key], rtol=1e-3, atol=1e-3
                        )
                    else:
                        assert torch.equal(reference[key], optimized[key])
                self.reference_checks.append(differences)
            self.initial_features = {
                k: v.detach().clone() for k, v in self.slow(0).items()
            }
        torch.cuda.synchronize()

    @torch.inference_mode()
    def slow(self, index):
        inputs = self.backbone_inputs[index % len(self.backbone_inputs)]
        if not self.features_only:
            return self.model.backbone(inputs)
        backbone = self.model.backbone.model
        selected = {
            k: inputs[k]
            for k in ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        }
        # HF ties inner hidden_states[-1] to the post-RMSNorm result. The outer
        # causal-LM recorder used by the released policy retains pre-norm output.
        # Capture exactly that tensor while skipping unused vocabulary logits.
        backbone.model(
            **selected, use_cache=False, output_hidden_states=False, return_dict=True
        )
        return BatchFeature(
            {
                "backbone_features": self._pre_final_norm,
                "backbone_attention_mask": selected["attention_mask"] == 1,
                "image_mask": selected["input_ids"] == backbone.config.image_token_id,
            }
        )

    @torch.inference_mode()
    def reset(self, seed=1234):
        torch.cuda.synchronize()
        torch.manual_seed(seed)
        self.model.action_head.reset_streaming_buffer()
        initial = self.model.action_head.get_action(
            BatchFeature(self.initial_features),
            BatchFeature(self.action_inputs[0]),
            options={"force_nonstreaming": True, "num_inference_timesteps": 4},
        )["action_pred"]
        self.model.action_head.seed_streaming_buffer(initial, slide_steps=1)
        torch.cuda.synchronize()

    @torch.inference_mode()
    def fast(self, features, index, image_delay=1):
        return self.model.forward_dit_action_only(
            BatchFeature(features),
            self.action_inputs[index % len(self.action_inputs)],
            options={
                "slide_steps": 1,
                "num_inference_steps_per_call": 1,
                "image_delay": image_delay,
            },
        )["action_pred"]

    def metadata(self):
        def describe(mapping):
            return {
                k: {
                    "shape": list(v.shape),
                    "dtype": str(v.dtype),
                    "device": str(v.device),
                }
                for k, v in mapping.items()
                if torch.is_tensor(v)
            }

        inputs = self.backbone_inputs[0]
        return {
            "precision": str(self.model.dtype),
            "action_horizon": self.model.config.action_horizon,
            "s1_nfe": 1,
            "slide_steps": 1,
            "samples": len(self.backbone_inputs),
            "features_only_s2": self.features_only,
            "features_only_reference_checks": self.reference_checks,
            "backbone_inputs": describe(
                {
                    k: inputs[k]
                    for k in [
                        "input_ids",
                        "attention_mask",
                        "pixel_values",
                        "image_grid_thw",
                    ]
                }
            ),
            "action_inputs": describe(self.action_inputs[0]),
            "cache": describe(self.initial_features),
        }
