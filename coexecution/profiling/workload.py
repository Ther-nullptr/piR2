"""Real LIBERO fixtures and fixed piR2 bootstrap, loaded only by the GPU worker."""

import hashlib
from contextlib import ExitStack

import torch

from .common import metadata, save, tensor_digest


class LiberoWorkload:
    def __init__(self, args):
        self.stack = ExitStack()
        try:
            self._load(args)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def close(self):
        self.stack.close()

    def _load(self, args):
        import numpy as np
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
        from transformers import BatchFeature

        from coexecution.libero.backend import Backend
        from coexecution.libero.service import load_core

        self.BatchFeature = BatchFeature
        self.policy = load_core(args.checkpoint, args.output, "cuda:0")
        from coexecution.libero.preprocessing import ImageResolutionScope, image_config

        self.image_resolution = self.stack.enter_context(
            ImageResolutionScope(self.policy, image_config(args))
        )
        self.model = self.policy.model.eval()
        self.head = self.model.action_head
        if not self.head.config.streaming or self.head.config.action_horizon != 40:
            raise ValueError("Need the H40 trained piR2 checkpoint")
        self.counts = {"vlm": 0, "dit": 0}
        self.hooks = []
        for label, module in (("vlm", self.model.backbone), ("dit", self.head.model)):

            def count(_module, _inputs, _output, label=label):
                self.counts[label] += 1

            handle = module.register_forward_hook(count)
            self.hooks.append(handle)
            self.stack.callback(handle.remove)
        modalities = {
            key: value
            for key, value in self.policy.modality_configs.items()
            if key != "action"
        }
        loader = LeRobotEpisodeLoader(
            args.dataset, modalities, video_backend="torchcodec"
        )
        trajectory = loader[args.episode_position]
        self.backbone_inputs, self.action_inputs, self.fixtures = [], [], []

        def observation(step):
            sample = extract_step_data(
                trajectory, step, modalities, self.policy.embodiment_tag
            )
            return {
                "state": {
                    key: np.asarray(value, dtype=np.float32)[None]
                    for key, value in sample.states.items()
                },
                "video": {
                    key: np.asarray(value)[None] for key, value in sample.images.items()
                },
                "language": {
                    key: [[sample.text]] for key in modalities["language"].modality_keys
                },
            }, sample.text

        bootstrap_observation = None
        for index in range(args.frames):
            image_step = args.dataset_step + index
            visual, text = observation(image_step)
            state, _ = observation(image_step + 3)
            if bootstrap_observation is None:
                bootstrap_observation = visual
            collated, _ = self.policy._collate_observation(visual)
            backbone, _ = self.model.prepare_input(collated["inputs"])
            action, _ = self.policy._collate_state_only({"state": state["state"]})
            action = {
                key: value.to(
                    device="cuda:0",
                    dtype=self.model.dtype
                    if value.is_floating_point()
                    else value.dtype,
                )
                for key, value in action.items()
            }
            self.backbone_inputs.append(backbone)
            self.action_inputs.append(action)
            arrays = {f"video.{key}": value for key, value in visual["video"].items()}
            arrays.update(
                {f"state.{key}": value for key, value in state["state"].items()}
            )
            np.savez_compressed(args.output / f"fixture-{index:02d}.npz", **arrays)
            self.fixtures.append(
                {
                    "index": index,
                    "episode_position": args.episode_position,
                    "image_dataset_step": image_step,
                    "state_dataset_step": image_step + 3,
                    "image_delay_ticks": 3,
                    "text": text,
                    "raw_sha256": {
                        key: hashlib.sha256(value.tobytes()).hexdigest()
                        for key, value in arrays.items()
                    },
                    "backbone_inputs": metadata(backbone),
                    "action_inputs": metadata(action),
                }
            )
        # Common unfused BF16 bootstrap, identical across all precision conditions.
        backend = Backend(self.policy, "pir2", args.output)
        with torch.inference_mode(), torch.cuda.stream(backend.stream):
            boot = backend.bootstrap(bootstrap_observation, 1000.0, 1, args.seed)
        backend.stream.synchronize()
        if boot["forward_counts"] != {"vlm": 1, "dit": 4}:
            raise RuntimeError(
                "Bootstrap must run exactly one VLM and four DiT forwards"
            )
        self.buffer = self.head._stream_buf.clone()
        self.times = self.head._stream_buf_t.clone()
        self.rng = torch.cuda.get_rng_state()
        torch.cuda.synchronize()
        self.bootstrap = {
            "precision": "bf16",
            "fusion": False,
            "forward_counts": boot["forward_counts"],
            "buffer_sha256": tensor_digest(self.buffer),
            "times_sha256": tensor_digest(self.times),
            "rng_sha256": tensor_digest(self.rng),
            "seed": args.seed,
            "excluded_from_measurement": True,
        }
        save(args.output / "fixtures.json", self.fixtures)
        save(args.output / "bootstrap.json", self.bootstrap)

    def restore(self):
        self.head._stream_buf = self.buffer.clone()
        self.head._stream_buf_t = self.times.clone()
        torch.cuda.set_rng_state(self.rng)
        torch.cuda.current_stream().synchronize()

    def vlm(self, index):
        features = self.model.backbone(self.backbone_inputs[index])
        if self.image_resolution.config.size is not None:
            self.image_resolution.record_features(features)
        return features

    def dit(self, features, index):
        return self.model.forward_dit_action_only(
            self.BatchFeature(data={**features}),
            self.action_inputs[index],
            {"slide_steps": 1, "num_inference_steps_per_call": 1, "image_delay": 3},
        )["action_pred"]
