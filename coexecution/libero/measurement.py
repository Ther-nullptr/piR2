"""Paired CUDA LIBERO policy timing with explicit, independent boundaries.

The raw policy, preparation/transfer and prepared-model paths are separate
measurements. They are not an additive wall-time decomposition. Import/CLI help
loads no neural framework. Models and numerical model checks require CUDA.
"""

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODES = ("baseline", "prepared", "tokenized", "static", "graph", "images", "compiled")
RUNTIME_FIELDS = (
    "input_preparation",
    "tokenizer_cache",
    "static_vlm",
    "vlm_graph",
    "image_array_preparation",
    "compiled_host_guards",
)


def reference_mode(modes):
    """Keep baseline as reference when requested, otherwise use the first mode."""
    if not modes or len(set(modes)) != len(modes) or set(modes) - set(MODES):
        raise ValueError("Unique known measurement modes are required")
    return "baseline" if "baseline" in modes else modes[0]


def runtime_settings(mode):
    if mode not in MODES:
        raise ValueError(f"Unknown runtime mode: {mode}")
    level = MODES.index(mode)
    return {name: level > index for index, name in enumerate(RUNTIME_FIELDS)}


def forward_prepared(model, backbone_inputs, action_inputs, options=None):
    """The pinned GR00T get_action body after prepare_input, unchanged."""
    features = model.backbone(backbone_inputs)
    return model.action_head.get_action(features, action_inputs, options)


def round_plan(modes, observations, rounds, seed):
    if not modes or len(set(modes)) != len(modes) or observations < 1 or rounds < 1:
        raise ValueError("Unique modes and positive observations/rounds are required")
    rng = random.Random(seed)
    rows = []
    for index in range(rounds):
        order, samples = list(modes), list(range(observations))
        rng.shuffle(order)
        rng.shuffle(samples)
        rows.append(
            {
                "round": index,
                "modes": order,
                "observations": samples,
                "seeds": [rng.randrange(2**31) for _ in samples],
            }
        )
    return rows


def action_difference(reference, candidate):
    import numpy as np

    if reference.keys() != candidate.keys():
        return {"passes": False, "reason": "action keys differ", "max_abs": None}
    maximum = 0.0
    for key in reference:
        left, right = np.asarray(reference[key]), np.asarray(candidate[key])
        if left.shape != right.shape:
            return {"passes": False, "reason": "action shapes differ", "max_abs": None}
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            return {"passes": False, "reason": "nonfinite actions", "max_abs": None}
        if left.size:
            maximum = max(maximum, float(np.abs(left.astype(float) - right).max()))
    return {"passes": maximum == 0.0, "max_abs": maximum, "reason": None}


def summarize_samples(rows):
    import numpy as np

    grouped = {}
    for row in rows:
        grouped.setdefault(row["mode"], {}).setdefault(row["scope"], []).append(
            row["wall_ms"]
        )
    return {
        mode: {
            scope: {
                "samples": len(values),
                "mean_ms": statistics.mean(values),
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
                "min_ms": min(values),
                "max_ms": max(values),
            }
            for scope, values in scopes.items()
        }
        for mode, scopes in grouped.items()
    }


def graph_interval(before, after, *, expected_replays):
    """Reject cold capture, failures or silent eager bypass in timed calls."""
    if before is None and after is None:
        return {"enabled": False, "passes": True}
    if before is None or after is None:
        return {"enabled": True, "passes": False, "reason": "Graph scope changed"}
    deltas = {
        name + "_delta": after[name + "_count"] - before[name + "_count"]
        for name in ("capture", "replay", "failure")
    }
    result = {
        "enabled": True,
        "passes": deltas
        == {"capture_delta": 0, "replay_delta": expected_replays, "failure_delta": 0},
        "expected_replays": expected_replays,
        **deltas,
    }
    if "logical_nfe_count" in before or "logical_nfe_count" in after:
        logical = after.get("logical_nfe_count", 0) - before.get("logical_nfe_count", 0)
        result["logical_nfe_delta"] = logical
        result["passes"] = result["passes"] and logical == 4 * expected_replays
    return result


def graph_snapshot(session):
    dit = None if session.graphs is None else session.graphs.evidence()
    vlm = None
    if getattr(session.config, "vlm_graph", False):
        if session.static_vlm is None:
            raise RuntimeError("VLM graph enabled without an installed adapter")
        vlm = session.static_vlm.evidence()["vlm_graph"]
    action = getattr(session, "action_graph", None)
    return {
        "dit": dit,
        "vlm": vlm,
        "action": None if action is None else action.evidence(),
    }


def complete_graph_interval(before, after):
    result = {
        name: graph_interval(before.get(name), after.get(name), expected_replays=count)
        for name, count in (("dit", 4), ("vlm", 1), ("action", 1))
    }
    return {"passes": all(value["passes"] for value in result.values()), **result}


def build_parser():
    from coexecution.optimization.config import add_optimization_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "dataset", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precisions",
        choices=("bf16", "w8a8", "w4a4"),
        nargs="+",
        default=["bf16", "w8a8", "w4a4"],
    )
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES[:5]))
    parser.add_argument("--observations", type=int, default=3)
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--profile-rounds", type=int, default=0)
    parser.add_argument("--start-step", type=int, default=10)
    parser.add_argument("--step-stride", type=int, default=7)
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--seed", type=int, default=127)
    parser.add_argument("--torch-threads", type=int, default=2)
    add_optimization_arguments(parser)
    return parser


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _source_manifest(output):
    names = [
        p
        for p in (ROOT / "coexecution").rglob("*.py")
        if not p.name.startswith("test_")
    ]
    names += list((ROOT / "coexecution" / "csrc").glob("*.cu"))
    result = {}
    for source in sorted(names):
        relative = source.relative_to(ROOT)
        data = source.read_bytes()
        target = output / "runtime-source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        result[str(relative)] = hashlib.sha256(data).hexdigest()
    _write(output / "runtime-source" / "manifest.json", result)
    return result


def _observations(policy, args):
    import numpy as np
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data

    modalities = {
        key: value for key, value in policy.modality_configs.items() if key != "action"
    }
    loader = LeRobotEpisodeLoader(
        args.dataset, modalities, video_backend=args.video_backend
    )
    episodes = args.episodes
    if episodes is None:
        episodes, seen = [], set()
        for index, meta in enumerate(loader.episodes_metadata):
            instruction = tuple(meta.get("tasks", []))
            if instruction not in seen:
                episodes.append(index)
                seen.add(instruction)
            if len(episodes) >= args.observations:
                break
    if not episodes:
        raise ValueError("No LIBERO observations selected")
    data, observations, identity = {}, [], []
    for index in range(args.observations):
        episode = episodes[index % len(episodes)]
        if episode not in data:
            data[episode] = loader[episode]
        step = args.start_step + index // len(episodes) * args.step_stride
        extracted = extract_step_data(
            data[episode], step, modalities, policy.embodiment_tag
        )
        observation = {
            "state": {
                key: np.asarray(value, dtype=np.float32)[None]
                for key, value in extracted.states.items()
            },
            "video": {
                key: np.asarray(value)[None] for key, value in extracted.images.items()
            },
            "language": {
                key: [[extracted.text]] for key in modalities["language"].modality_keys
            },
        }
        policy.check_observation(observation)
        digest = hashlib.sha256(
            json.dumps(observation["language"], sort_keys=True).encode()
        )
        arrays = {}
        for kind in ("state", "video"):
            arrays[kind] = {}
            for key, value in sorted(observation[kind].items()):
                digest.update(f"{kind}/{key}/{value.shape}/{value.dtype}".encode())
                digest.update(value.tobytes())
                arrays[kind][key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
        observations.append(observation)
        identity.append(
            {
                "episode": episode,
                "step": step,
                "sha256": digest.hexdigest(),
                "arrays": arrays,
                "language": observation["language"],
            }
        )
    return observations, identity


def _config(args, precision, mode):
    from coexecution.optimization.config import optimization_config

    values = dict(vars(args), inference_precision=precision)
    if precision == "bf16":
        for key in (
            "group_conditioning",
            "norm_modulation_quant",
            "residual_norm_quant",
        ):
            values[key] = False
    base = optimization_config(argparse.Namespace(**values))
    supported = {field.name for field in fields(base)}
    settings = runtime_settings(mode)
    missing = [
        name for name, enabled in settings.items() if enabled and name not in supported
    ]
    if missing:
        raise RuntimeError(f"Runtime mode {mode} needs uninstalled adapters: {missing}")
    return replace(
        base, **{name: value for name, value in settings.items() if name in supported}
    )


def _seed_reset(policy, seed, device):
    import torch

    policy.reset_streaming_buffer()
    torch.manual_seed(seed)
    with torch.cuda.device(device):
        torch.cuda.manual_seed(seed)


def _prepare(policy, observation):
    collated, states = policy._collate_observation(observation)
    return (*policy.model.prepare_input(collated["inputs"]), states)


def _timed(function, device):
    import torch

    torch.cuda.synchronize(device)
    start = time.perf_counter_ns()
    output = function()
    torch.cuda.synchronize(device)
    return output, (time.perf_counter_ns() - start) / 1e6


def timed_model(function, device):
    """CUDA span includes submission gaps; host return is elapsed, not CPU work."""
    import torch

    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    torch.cuda.synchronize(device)
    begin.record()
    start = time.perf_counter_ns()
    output = function()
    host_return = (time.perf_counter_ns() - start) / 1e6
    end.record()
    torch.cuda.synchronize(device)
    wall = (time.perf_counter_ns() - start) / 1e6
    return (
        output,
        wall,
        {
            "cuda_span_ms": begin.elapsed_time(end),
            "host_return_ms": host_return,
        },
    )


def run(args):
    reference_name = reference_mode(args.modes)
    import torch

    if args.upstream is not None:
        sys.path.insert(0, str(args.upstream))
    from coexecution.libero.service import load_core
    from coexecution.optimization.session import GrootOptimizations
    from scripts.libero_experiment_utils import checkpoint_identity

    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("The complete LIBERO model measurement requires CUDA")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = _source_manifest(args.output)
    configs = {
        f"{precision}-{mode}": _config(args, precision, mode)
        for precision in args.precisions
        for mode in args.modes
    }
    policy = load_core(args.checkpoint, args.output, args.device)
    if policy.model.config.streaming or policy.model.config.action_horizon != 40:
        raise ValueError(
            "This replay requires the official non-streaming Spatial H40 model"
        )
    if policy.model.action_head.num_inference_timesteps != 4:
        raise ValueError("Expected unchanged Flow4 NFE=4")
    observations, identity = _observations(policy, args)
    _write(args.output / "observations.json", identity)
    plan = round_plan(args.modes, len(observations), args.rounds, args.seed)
    _write(args.output / "round-plan.json", plan)
    samples, gates, evidence, graph_audits, profiles = [], [], {}, [], {}
    retained = {}
    import numpy as np

    def retain(label, actions):
        for key, value in actions.items():
            retained[f"{label}/{key}"] = value.copy()

    with (
        torch.cuda.device(args.device),
        torch.inference_mode(),
        (args.output / "samples.jsonl").open("w") as stream,
    ):
        for precision in args.precisions:
            reference_label = f"{precision}-{reference_name}"
            with GrootOptimizations(policy, configs[reference_label]) as optimized:
                setter = getattr(optimized, "set_runtime", None)
                if not callable(setter):
                    raise TypeError(
                        "Resident paired measurement requires GrootOptimizations.set_runtime"
                    )
                for round_row in plan:
                    outputs = {}
                    for mode in round_row["modes"]:
                        label = f"{precision}-{mode}"
                        setter(**runtime_settings(mode))
                        for _ in range(args.warmup):
                            for index, observation in enumerate(observations):
                                _seed_reset(
                                    policy, round_row["seeds"][index], args.device
                                )
                                policy.get_action(observation)
                        outputs[mode] = {}
                        for index in round_row["observations"]:
                            observation, seed = (
                                observations[index],
                                round_row["seeds"][index],
                            )
                            _seed_reset(policy, seed, args.device)
                            before = graph_snapshot(optimized)
                            (actions, _), wall = _timed(
                                lambda obs=observation: policy.get_action(obs),
                                args.device,
                            )
                            after_policy = graph_snapshot(optimized)
                            outputs[mode][index] = actions
                            retain(
                                f"r{round_row['round']}/{label}/o{index}/policy",
                                actions,
                            )
                            row = {
                                "round": round_row["round"],
                                "mode": label,
                                "scope": "policy",
                                "observation": index,
                                "seed": seed,
                                "wall_ms": wall,
                            }
                            samples.append(row)
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            # Independent probes: preparation executes once outside
                            # the prepared-model boundary. The neural model still
                            # recomputes the full VLM and every Flow step.
                            _seed_reset(policy, seed, args.device)
                            prepared, prep_wall = _timed(
                                lambda obs=observation: _prepare(policy, obs),
                                args.device,
                            )
                            backbone, action, states = prepared
                            prediction, model_wall, model_detail = timed_model(
                                lambda bb=backbone, act=action: forward_prepared(
                                    policy.model, bb, act
                                ),
                                args.device,
                            )
                            after_model = graph_snapshot(optimized)
                            audit = {
                                "round": round_row["round"],
                                "mode": label,
                                "observation": index,
                                "policy": complete_graph_interval(before, after_policy),
                                "prepared_model": complete_graph_interval(
                                    after_policy, after_model
                                ),
                            }
                            graph_audits.append(audit)
                            if not all(
                                audit[scope]["passes"]
                                for scope in ("policy", "prepared_model")
                            ):
                                _write(
                                    args.output / "graph-timing-audit.json",
                                    graph_audits,
                                )
                                raise RuntimeError(
                                    f"Timed call was not steady Graph replay: {audit}"
                                )
                            prepared_actions = policy._decode_action(prediction, states)
                            gate = action_difference(actions, prepared_actions)
                            gates.append(
                                {
                                    "round": round_row["round"],
                                    "mode": label,
                                    "observation": index,
                                    "prepared_model": gate,
                                }
                            )
                            if not gate["passes"]:
                                _write(args.output / "validation.json", gates)
                                np.savez_compressed(
                                    args.output / "actions.npz", **retained
                                )
                                raise RuntimeError(
                                    f"Prepared model output differs: {label}/{index}: {gate}"
                                )
                            retain(
                                f"r{round_row['round']}/{label}/o{index}/prepared_model",
                                prepared_actions,
                            )
                            for scope, elapsed in (
                                ("preparation_transfer", prep_wall),
                                ("prepared_model", model_wall),
                            ):
                                row = {
                                    "round": round_row["round"],
                                    "mode": label,
                                    "scope": scope,
                                    "observation": index,
                                    "seed": seed,
                                    "wall_ms": elapsed,
                                }
                                if scope == "prepared_model":
                                    row.update(model_detail)
                                samples.append(row)
                                stream.write(json.dumps(row, allow_nan=False) + "\n")
                        evidence[label] = optimized.evidence()
                    reference = outputs[reference_name]
                    for mode, candidates in outputs.items():
                        for index, actions in candidates.items():
                            gate = action_difference(reference[index], actions)
                            gates.append(
                                {
                                    "round": round_row["round"],
                                    "mode": f"{precision}-{mode}",
                                    "reference_mode": reference_label,
                                    "observation": index,
                                    "same_precision": gate,
                                }
                            )
                            if not gate["passes"]:
                                _write(args.output / "validation.json", gates)
                                np.savez_compressed(
                                    args.output / "actions.npz", **retained
                                )
                                raise RuntimeError(
                                    f"Runtime output differs from {reference_label}: {precision}/{mode}: {gate}"
                                )
                    stream.flush()
                    _write(args.output / "validation.json", gates)
                    _write(args.output / "graph-timing-audit.json", graph_audits)
                    print(
                        f"{precision}: round {round_row['round'] + 1}/{args.rounds} complete",
                        flush=True,
                    )
                if args.profile_rounds:
                    from coexecution.libero.profiling import collect_model_profile

                    for mode in args.modes:
                        label = f"{precision}-{mode}"
                        setter(**runtime_settings(mode))
                        profiles[label] = collect_model_profile(
                            policy,
                            observations,
                            optimized,
                            args.output / "profiles" / f"{label}.trace.json",
                            rounds=args.profile_rounds,
                            seed=args.seed,
                            device=args.device,
                        )
                        _write(args.output / "profiles.json", profiles)
    for name, sha in sources.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise RuntimeError(f"Measured source changed: {name}")
    result = {
        "scope": "Official Spatial local policy replay; RPC/simulator/AoI are separate deployment measurements",
        "boundaries": {
            "policy": "Raw CPU RGB/state/language -> decoded CPU action, CUDA complete",
            "preparation_transfer": "Raw observation -> collated CUDA input trees, CUDA complete",
            "prepared_model": "Prepared resident inputs -> full VLM + Flow4 CUDA action, CUDA complete",
        },
        "nonadditive": True,
        "timing_notes": (
            "Prepared-model CUDA span includes device idle/submission gaps; "
            "host_return_ms overlaps GPU work and can include synchronization. "
            "Neither is a CPU-work sum or a profiler kernel-work sum."
        ),
        "profiler": False,
        "checkpoint": checkpoint_identity(args.checkpoint),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "device": {
            "name": torch.cuda.get_device_name(args.device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "runtime_sources": sources,
        "reference_mode": reference_name,
        "observations": identity,
        "latency": summarize_samples(samples),
        "mode_evidence": evidence,
        "validation": gates,
        "graph_timing_audit": graph_audits,
        "profiles": profiles,
    }
    np.savez_compressed(args.output / "actions.npz", **retained)
    _write(args.output / "results.json", result)
    print(json.dumps(result["latency"], indent=2), flush=True)
    return result


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        reference_mode(args.modes)
    except ValueError as error:
        parser.error(str(error))
    if any(
        getattr(args, name) < 1
        for name in ("observations", "rounds", "warmup", "step_stride", "torch_threads")
    ):
        parser.error("Observation/round/warmup/stride/thread counts must be positive")
    if (
        args.start_step < 0
        or args.profile_rounds < 0
        or len(set(args.precisions)) != len(args.precisions)
    ):
        parser.error("Nonnegative start step and unique precisions are required")
    run(args)


if __name__ == "__main__":
    main()
