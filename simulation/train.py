"""Train an independent Leap flow policy on GPU with paper optimizer settings."""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from simulation.model import FlowUNet, schedule


def prepare_data(path, device):
    data = np.load(path)
    lengths = data["lengths"]
    observations, actions = data["observations"], data["actions"]
    valid_obs = np.concatenate([x[:n] for x, n in zip(observations, lengths)])
    valid_actions = np.concatenate([x[:n] for x, n in zip(actions, lengths)])
    stats = {}
    for key, values in [("obs", valid_obs), ("action", valid_actions)]:
        lower, upper = values.min(0), values.max(0)
        stats[key + "_offset"] = (lower + upper) / 2
        stats[key + "_scale"] = np.maximum((upper - lower) / 2, 1e-3)
    obs_windows, action_windows = [], []
    for obs, act, length in zip(observations, actions, lengths):
        t = np.arange(length)
        obs_index = np.maximum(t[:, None] + np.array([-1, 0]), 0)
        action_index = np.minimum(t[:, None] + np.arange(16), length - 1)
        obs_windows.append((obs[obs_index] - stats["obs_offset"]) / stats["obs_scale"])
        action_windows.append(
            (act[action_index] - stats["action_offset"]) / stats["action_scale"]
        )
    obs = torch.tensor(np.concatenate(obs_windows), device=device)
    act = torch.tensor(np.concatenate(action_windows), device=device)
    return obs, act, {k: v.tolist() for k, v in stats.items()}


def main(args):
    assert torch.cuda.is_available(), "CUDA is required"
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rng = np.random.default_rng(args.seed)
    data_generator = torch.Generator(device="cuda").manual_seed(args.seed)
    obs, actions, stats = prepare_data(args.data, "cuda")
    model = FlowUNet(dims=tuple(args.dims)).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        fused=True,
    )
    batches = math.ceil(len(obs) / args.batch_size)
    total_steps = batches * args.epochs
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "data": str(args.data),
        "output": str(args.output),
        "resume": str(args.resume) if args.resume else None,
    }
    config.update(
        {
            "frames": len(obs),
            "parameters": sum(p.numel() for p in model.parameters()),
            "normalization": "training-set min/max to [-1,1]",
            "ema": False,
        }
    )
    data_sha = hashlib.sha256(args.data.read_bytes()).hexdigest()
    provenance_path = args.output / "data-manifest.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if provenance["sha256"] != data_sha:
            raise ValueError("Training dataset content changed since the run began")
    elif args.resume:
        raise ValueError("Cannot verify resume data without data-manifest.json")
    else:
        provenance_path.write_text(
            json.dumps({"path": str(args.data.resolve()), "sha256": data_sha}, indent=2)
        )
    config["data_sha256"] = data_sha
    started = time.monotonic()
    global_step = 0
    start_epoch = 1
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        for key in ["method", "seed", "dims", "batch_size", "epochs"]:
            if saved["config"][key] != config[key]:
                raise ValueError(f"Resume configuration changed: {key}")
        if saved["stats"] != stats or saved["config"]["frames"] != len(obs):
            raise ValueError("Resume normalization or dataset frame count changed")
        if saved["config"].get("data_sha256", data_sha) != data_sha:
            raise ValueError("Resume checkpoint dataset fingerprint differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["numpy_rng"]
        data_generator.set_state(saved["data_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        global_step = saved["global_step"]
        start_epoch = saved["epoch"] + 1
        print("Resuming from epoch", saved["epoch"], flush=True)
        del saved
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config), flush=True)
    metrics = (args.output / "training.jsonl").open("a", buffering=1)
    for epoch in range(start_epoch, args.epochs + 1):
        order = torch.randperm(len(obs), device="cuda", generator=data_generator)
        losses = torch.zeros((), device="cuda")
        for ids in order.split(args.batch_size):
            batch_obs, target = obs[ids], actions[ids]
            batch, horizon, _ = target.shape
            keep = torch.ones(batch, horizon, device="cuda")
            if args.method == "flow" or (args.method == "pir2" and rng.random() < 0.2):
                times = torch.rand(batch, 1, device="cuda").expand(-1, horizon)
            elif args.method == "rtc":
                delay = int(rng.integers(0, 11))
                times = torch.rand(batch, 1, device="cuda").expand(-1, horizon).clone()
                times[:, :delay] = 1
                keep[:, :delay] = 0
            else:
                delay = int(rng.integers(1, 6))
                base = schedule(horizon, delay, "cuda")
                jitter = (torch.rand(batch, 1, device="cuda") * 2 - 1) * (
                    0.25 / (horizon - 2 * delay)
                )
                times = (base[None] + jitter).clamp(0, 1)
                times[:, :delay] = 1
                keep[:, :delay] = 0
            noise = torch.randn_like(target)
            noisy = (1 - times[..., None]) * noise + times[..., None] * target
            warmup = min((global_step + 1) / 500, 1)
            progress = max(global_step - 500, 0) / max(total_steps - 500, 1)
            lr = 1e-4 * warmup * 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                velocity = model(noisy, times, batch_obs)
                loss = (
                    ((velocity.float() - (target - noise)) ** 2) * keep[..., None]
                ).sum()
                loss = loss / (keep.sum() * target.shape[-1])
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses += loss.detach()
            global_step += 1
        if not torch.isfinite(losses):
            raise RuntimeError("Non-finite training loss")
        row = {
            "epoch": epoch,
            "step": global_step,
            "loss": (losses / batches).item(),
            "grad_norm": float(grad),
            "lr": lr,
            "elapsed_seconds": time.monotonic() - started,
        }
        metrics.write(json.dumps(row) + "\n")
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(row), flush=True)
        checkpoint = {
            "model": model.state_dict(),
            "stats": stats,
            "config": config,
            "epoch": epoch,
            "global_step": global_step,
        }
        latest = {
            **checkpoint,
            "optimizer": optimizer.state_dict(),
            "numpy_rng": rng.bit_generator.state,
            "data_rng": data_generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }
        torch.save(latest, args.output / "latest.tmp")
        os.replace(args.output / "latest.tmp", args.output / "latest.pt")
        if epoch == 1 or epoch % args.save_every == 0 or epoch == args.epochs:
            target_path = args.output / f"epoch-{epoch:04d}.pt"
            temporary = target_path.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, target_path)
            print("Saved", target_path, flush=True)
    metrics.close()
    # If a restart found the final latest.pt after interruption during snapshot
    # writing, materialize the final inference checkpoint before reporting success.
    final_path = args.output / f"epoch-{args.epochs:04d}.pt"
    if not final_path.exists():
        torch.save(
            {
                "model": model.state_dict(),
                "stats": stats,
                "config": config,
                "epoch": args.epochs,
                "global_step": global_step,
            },
            final_path.with_suffix(".tmp"),
        )
        os.replace(final_path.with_suffix(".tmp"), final_path)
    print("TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("outputs/simulation/demonstrations.npz")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["flow", "rtc", "pir2"], default="pir2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dims", type=int, nargs=3, default=[256, 512, 1024])
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    main(parser.parse_args())
