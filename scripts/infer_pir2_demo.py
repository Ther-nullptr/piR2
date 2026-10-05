"""Run the official streaming policy on real SO100 demo observations using CUDA."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.decoupled_policy import DecoupledGr00tPolicy

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "upstream/learning/Isaac-GR00T"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--slide-steps", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/pir2-inference.json"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script does not run on CPU")
    if args.calls < 1 or args.slide_steps < 1:
        parser.error("calls and slide-steps must be positive")

    torch.manual_seed(0)
    tag = EmbodimentTag.NEW_EMBODIMENT
    policy = DecoupledGr00tPolicy(tag, str(args.checkpoint), device="cuda:0")
    if not policy.model.config.streaming:
        raise ValueError("Checkpoint must have streaming enabled by piR2 training")
    horizon = policy.model.config.action_horizon
    if 2 * args.slide_steps >= horizon:
        parser.error("slide-steps must be less than half the action horizon")
    loader = LeRobotEpisodeLoader(
        dataset_path=REPO / "demo_data/cube_to_bowl_5",
        modality_configs=policy.modality_configs,
        video_backend="torchcodec",
    )
    trajectory = loader[0]
    modalities = deepcopy(policy.modality_configs)
    modalities.pop("action", None)

    def observation(step):
        data = extract_step_data(trajectory, step, modalities, tag)
        return {
            "state": {
                key: np.asarray(value)[None] for key, value in data.states.items()
            },
            "video": {
                key: np.asarray(value)[None] for key, value in data.images.items()
            },
            "language": {
                key: [[data.text]] for key in modalities["language"].modality_keys
            },
        }

    counts = {"vlm": 0, "dit": 0}

    def count_vlm(_module, _inputs, _output):
        counts["vlm"] += 1

    def count_dit(_module, _inputs, _output):
        counts["dit"] += 1

    handles = [
        policy.model.backbone.register_forward_hook(count_vlm),
        policy.model.action_head.model.register_forward_hook(count_dit),
    ]
    results = {
        "checkpoint": str(args.checkpoint.resolve()),
        "device": torch.cuda.get_device_name(0),
        "dataset": "cube_to_bowl_5, episode 0",
        "scope": "GPU replay inference; not closed-loop robot success evaluation",
        "calls": [],
    }
    initial, warmup = policy.seed_streaming_from_obs(
        observation(0),
        num_inference_timesteps=4,
        t_image_capture=0.0,
        slide_steps=args.slide_steps,
    )
    results["warmup"] = {**warmup, "forward_counts": dict(counts)}
    if not all(np.isfinite(value).all() for value in initial.values()):
        raise RuntimeError("Warm-start produced non-finite actions")

    for call in range(args.calls):
        step = (call + 1) * args.slide_steps
        obs = observation(step)
        # Refresh the slow channel every second call, always refresh proprioception.
        if call % 2 == 0:
            policy.update_vlm_cache(obs, t_image_capture=step / 30.0)
        before = dict(counts)
        actions, info = policy.get_action_chunk_cached(
            {"state": obs["state"]},
            options={
                "slide_steps": args.slide_steps,
                "num_inference_steps_per_call": 1,
            },
            t_state_capture=step / 30.0,
            period_ms=1000.0 / 30.0,
        )
        delta = {key: counts[key] - before[key] for key in counts}
        if delta != {"vlm": 0, "dit": 1}:
            raise RuntimeError(f"Unexpected cached-call forward counts: {delta}")
        if not all(np.isfinite(value).all() for value in actions.values()):
            raise RuntimeError("Streaming inference produced non-finite actions")
        results["calls"].append(
            {
                "step": step,
                "forward_counts": delta,
                "shapes": {key: list(value.shape) for key, value in actions.items()},
                "new_actions": {
                    key: value[0, args.slide_steps : 2 * args.slide_steps].tolist()
                    for key, value in actions.items()
                },
                "cache_id": info.get("cache_id"),
                "image_delay_ticks": info.get("image_delay_ticks"),
            }
        )
        print(f"Completed streaming call {call + 1}/{args.calls}", flush=True)

    for handle in handles:
        handle.remove()
    results["all_actions_finite"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
