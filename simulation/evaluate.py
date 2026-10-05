"""Measure success rates with explicit action/vision delays in GPU Leap rollouts."""

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from simulation.leap_env import PaperLeap, make_advance
from simulation.model import FlowUNet, clamp_action, flow_sample, schedule, stream_step


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return [centre - half, centre + half]


def main(args):
    assert torch.cuda.is_available() and jax.default_backend() == "gpu"
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = FlowUNet(dims=tuple(config["dims"])).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    del checkpoint["model"]
    stats = {k: torch.tensor(v, device="cuda") for k, v in checkpoint["stats"].items()}
    env = PaperLeap(batch_size=args.episodes)
    reset = jax.jit(jax.vmap(env.reset))
    advance = make_advance(env, args.episodes)

    result = {
        "checkpoint": str(args.checkpoint),
        "epoch": checkpoint["epoch"],
        "method": config["method"],
        "training_seed": config["seed"],
        "evaluation_seed": args.seed,
        "episodes_per_cell": args.episodes,
        "control_hz": 50,
        "max_steps": 600,
        "threshold_rad": 0.2,
        "physics_step_wrapper": "scan_length_one",
        "scope": "Independent reconstruction; public single-expert demonstrations",
        "inference_mode": "full_flow_sync_diagnostic"
        if args.full_flow_sync
        else "checkpoint",
        "cells": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def predict(noisy, times, observations):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(noisy, times, observations).float()

    for d0 in args.d0:
        torch.manual_seed(args.seed)
        method = config["method"]
        delay = (
            (d0 if args.no_async else 1) if method == "pir2" else math.ceil(1.75 * d0)
        )
        vision_delay = d0 if method == "pir2" and not args.no_async else 0
        if args.full_flow_sync:
            delay, vision_delay = 1, 0
        started = time.monotonic()
        state = reset(jax.random.split(jax.random.PRNGKey(args.seed), args.episodes))
        jax.block_until_ready(state)
        raw = torch.from_dlpack(state.obs["policy_state"])
        raw_history = deque(
            [raw.clone() for _ in range(vision_delay + 1)], maxlen=vision_delay + 1
        )
        obs_history = deque([raw.clone(), raw.clone()], maxlen=2)

        def observations(obs_history=obs_history):
            return (
                torch.stack(list(obs_history), dim=1) - stats["obs_offset"]
            ) / stats["obs_scale"]

        initial = flow_sample(predict, observations(), steps=15)
        prefix = initial[:, :delay]
        if method == "pir2":
            times = schedule(16, delay, "cuda").expand(args.episodes, -1)
            buffer = (1 - times[..., None]) * torch.randn_like(initial) + times[
                ..., None
            ] * initial
        alive = jnp.ones(args.episodes, bool)
        successes = jnp.zeros(args.episodes, bool)
        lengths = jnp.zeros(args.episodes, jnp.int32)
        for tick in range(0, 600, delay):
            with torch.no_grad():
                if args.full_flow_sync:
                    prediction = flow_sample(predict, observations(), steps=15)
                    executing = prediction[:, :1]
                elif method == "pir2":
                    buffer[:, :delay] = clamp_action(
                        buffer[:, :delay], stats["action_offset"], stats["action_scale"]
                    )
                    prediction, buffer, times = stream_step(
                        predict, buffer, times, observations(), delay
                    )
                    executing = prediction[:, :delay]
                else:
                    prefix = clamp_action(
                        prefix, stats["action_offset"], stats["action_scale"]
                    )
                    prediction = flow_sample(
                        predict,
                        observations(),
                        steps=15,
                        prefix=prefix if method == "rtc" else None,
                    )
                    executing = prefix
                    prefix = prediction[:, delay : 2 * delay]
                physical_actions = (
                    executing * stats["action_scale"] + stats["action_offset"]
                ).clamp(-1, 1)
            for offset in range(min(delay, 600 - tick)):
                action = jax.dlpack.from_dlpack(
                    physical_actions[:, offset].contiguous()
                )
                state, alive, successes, lengths = advance(
                    state, action, alive, successes, lengths
                )
                raw = torch.from_dlpack(state.obs["policy_state"])
                raw_history.append(raw)
                observed = torch.cat([raw[:, :32], raw_history[0][:, 32:]], dim=-1)
                obs_history.append(observed)
            if (tick // delay) % max(100 // delay, 1) == 0:
                print(
                    f"d0={d0}, tick={tick}, successes={int(successes.sum())}, alive={int(alive.sum())}",
                    flush=True,
                )
                if not bool(alive.any()):
                    break
        flags, episode_lengths = np.asarray(successes), np.asarray(lengths)
        count = int(flags.sum())
        cell = {
            "d0": d0,
            "action_delay": 0 if args.full_flow_sync else delay,
            "visual_delay": vision_delay,
            "successes": count,
            "episodes": args.episodes,
            "success_rate": count / args.episodes,
            "wilson_95_ci": wilson(count, args.episodes),
            "episode_successes": flags.tolist(),
            "episode_lengths": episode_lengths.tolist(),
            "elapsed_wall_seconds": time.monotonic() - started,
        }
        result["cells"].append(cell)
        args.output.write_text(json.dumps(result, indent=2))
        print(
            json.dumps({k: v for k, v in cell.items() if not k.startswith("episode_")}),
            flush=True,
        )
    print("EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--d0", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--no-async", action="store_true")
    parser.add_argument(
        "--full-flow-sync",
        action="store_true",
        help="Diagnostic: same weights, 15-NFE flow with fresh observations and no simulated latency",
    )
    main(parser.parse_args())
