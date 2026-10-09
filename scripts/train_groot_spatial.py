"""Matched-budget full-suite Flow or piR2 adaptation from public task weights."""

import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import torch
from gr00t.configs.base_config import get_default_config
from gr00t.configs.data.data_config import SingleDatasetConfig
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.experiment import experiment
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from transformers import TrainerCallback

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["pir2", "flow"], required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--stop-at", type=int, default=0)
    args = parser.parse_args()
    if 64 % args.microbatch:
        raise ValueError("Microbatch must divide effective batch64")
    delay = 5 if args.variant == "pir2" else 0
    os.environ["GR00T_IMAGE_DELAY_MAX"] = str(delay)
    output = ROOT / f"outputs/libero-{args.variant}-seed1000"
    run_dir = output / f"libero-{args.variant}-seed1000"
    resuming = any(run_dir.glob("checkpoint-*/trainer_state.json"))
    audit_dir = ROOT / f"artifacts/libero-pir2/{args.variant}-train"
    audit_dir.mkdir(parents=True, exist_ok=True)
    cfg = get_default_config()
    cfg.data.modality_configs = deepcopy(MODALITY_CONFIGS)
    cfg.data.modality_configs["libero_sim"]["action"].delta_indices = list(range(40))
    cfg.data.modality_configs["libero_sim"]["video"].delta_indices = [0]
    cfg.data.datasets = [
        SingleDatasetConfig([str(ROOT / "datasets/groot-libero-spatial")], "libero_sim")
    ]
    cfg.data.seed = 1000
    cfg.data.shard_size = 512
    cfg.data.episode_sampling_rate = 1.0
    cfg.data.num_shards_per_epoch = 104
    cfg.data.allow_padding = True
    cfg.data.override_pretraining_statistics = False
    cfg.model.action_horizon = 40
    cfg.model.streaming = args.variant == "pir2"
    cfg.model.streaming_constant_weight = 0.2
    cfg.model.streaming_chunk_wise_weight = 0.8
    cfg.model.streaming_schedule_mode = "pir2"
    cfg.model.streaming_chunk_size_max = 5
    cfg.model.streaming_mask_clean_end = True
    cfg.model.image_delay_max = delay
    cfg.model.image_delay_embed_dim = 64 if delay else 0
    cfg.model.tune_llm = False
    cfg.model.tune_visual = False
    cfg.model.tune_projector = True
    cfg.model.tune_diffusion_model = True
    cfg.model.tune_vlln = True
    cfg.model.state_dropout_prob = 0.2
    cfg.training.start_from_checkpoint = str(
        ROOT / "models/GR00T-N1.7-LIBERO/libero_spatial"
    )
    cfg.training.output_dir = str(output)
    cfg.training.experiment_name = f"libero-{args.variant}-seed1000"
    cfg.training.num_gpus = 1
    cfg.training.global_batch_size = 64
    # This upstream version does not divide global_batch_size by accumulation.
    cfg.training.batch_size = args.microbatch
    cfg.training.gradient_accumulation_steps = 64 // args.microbatch
    cfg.training.max_steps = args.steps
    cfg.training.learning_rate = 1e-4
    cfg.training.weight_decay = 1e-5
    cfg.training.warmup_ratio = 0.05
    cfg.training.optim = "adamw_torch_fused"
    cfg.training.save_steps = 500
    cfg.training.save_total_limit = 3
    cfg.training.save_only_model = False
    cfg.training.dataloader_num_workers = 4
    cfg.training.logging_steps = 10
    cfg.training.use_wandb = False
    cfg.training.transformers_local_files_only = True
    (audit_dir / "run-spec.json").write_text(
        json.dumps(
            {
                "variant": args.variant,
                "seed": 1000,
                "suite": "libero_spatial",
                "tasks": 10,
                "episodes": 432,
                "frames": 52970,
                "horizon": 40,
                "effective_batch": 64,
                "microbatch": args.microbatch,
                "accumulation": 64 // args.microbatch,
                "optimizer_updates": args.steps,
                "image_delay_max": delay,
                "trainable_scope": "action head/projectors; pretrained VLM stays active in inference",
                "source_checkpoint": cfg.training.start_from_checkpoint,
                "checkpoint_directory": str(run_dir),
            },
            indent=2,
        )
        + "\n"
    )
    original_forward = Gr00tN1d7ActionHead.forward
    calls = 0
    delays = set()

    def audited_forward(self, backbone_output, action_input):
        nonlocal calls
        calls += 1
        if calls <= 32:
            action = action_input["action"]
            mask = action_input["action_mask"]
            assert action.shape[1:] == (40, 132), action.shape
            assert torch.all(mask[:, :, :7] == 1) and torch.all(mask[:, :, 7:] == 0), (
                "Incomplete action supervision"
            )
            values = action_input.get("image_delay")
            if values is not None:
                delays.update(values.detach().cpu().tolist())
            if args.variant == "pir2" and calls == 1 and not resuming:
                assert torch.count_nonzero(self.delay_embedding.weight) == 0, (
                    "New delay embedding must start at zero"
                )
        result = original_forward(self, backbone_output, action_input)
        if calls <= 32:
            d = (
                getattr(self, "_last_clean_prefix_length", 0)
                if self.config.streaming
                else 0
            )
            if d:
                assert torch.count_nonzero(result["action_mask"][:, :d]) == 0
            record = {
                "microbatches_audited": calls,
                "action_shape": list(action_input["action"].shape),
                "raw_valid_action_dimensions_per_position": 7,
                "observed_image_delays": sorted(delays),
                "last_clean_prefix_length": d,
                "loss": float(result["loss"].detach()),
            }
            (audit_dir / "batch-audit.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
            if calls == 32 and delay and delays != set(range(delay + 1)):
                raise RuntimeError(
                    f"Visual delay sampling coverage incomplete: {delays}"
                )
        return result

    Gr00tN1d7ActionHead.forward = audited_forward
    original_trainer = experiment.Gr00tTrainer

    class ProgressCallback(TrainerCallback):
        def on_step_end(self, training_args, state, control, **kwargs):
            (audit_dir / "progress.json").write_text(
                json.dumps(
                    {
                        "optimizer_updates": state.global_step,
                        "target": args.steps,
                        "updated_at_unix": time.time(),
                    },
                    indent=2,
                )
                + "\n"
            )
            if args.stop_at and state.global_step >= args.stop_at:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_log(self, training_args, state, control, logs=None, **kwargs):
            with (audit_dir / "metrics.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "step": state.global_step,
                            "wall_time_unix": time.time(),
                            **(logs or {}),
                        }
                    )
                    + "\n"
                )

    class AuditedTrainer(original_trainer):
        def __init__(self, *pos, **kwargs):
            super().__init__(*pos, **kwargs)
            self.add_callback(ProgressCallback())

    experiment.Gr00tTrainer = AuditedTrainer
    print(
        "Starting full Spatial adaptation", args.variant, "global_batch=64", flush=True
    )
    from libero_model_loading import task_checkpoint_backbone

    with task_checkpoint_backbone():
        experiment.run(cfg)


if __name__ == "__main__":
    main()
