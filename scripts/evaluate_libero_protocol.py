"""Two separate clocks: controlled delays, or paced asynchronous deployment."""

import argparse
import fcntl
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from gr00t.policy.server_client import PolicyClient
from libero.libero import benchmark, get_libero_path
from libero_observations import batch_observation, write_video
from libero_protocol_scheduler import CommandTimeline, initial_frame_index
from libero_reproducible_env import LiberoEnv
from libero_streaming_client import nested_observation

KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def snapshot(observation):
    return {
        k: v if isinstance(v, str) else np.asarray(v).copy()
        for k, v in observation.items()
    }


def nested(observation):
    return nested_observation(batch_observation(observation))


def env_action(raw):
    return {
        "action." + key: np.array([raw[i]], dtype=np.float64)
        for i, key in enumerate(KEYS)
    }


def checked_env_step(env, action):
    before = float(env._env.env.sim.data.time)
    result = env.step(env_action(action))
    after = float(env._env.env.sim.data.time)
    if not np.isclose(after - before, 0.05, atol=1e-7, rtol=0):
        raise RuntimeError(f"Physical control step is {after - before}s, expected0.05s")
    return result


def wait_server(port):
    client = PolicyClient(host="127.0.0.1", port=port, timeout_ms=5000)
    deadline = time.monotonic() + 900
    while not client.ping():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Server{port} not ready")
        time.sleep(2)
    client.timeout_ms = 120000
    client._init_socket()
    return client


def metric_summary(values):
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {
        "count": len(a),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def algorithm_episode(env, observation, client, args, seed, trace):
    period = 1 / args.control_hz
    origin = 1000.0
    d = args.action_delay
    dv = args.visual_delay
    first = snapshot(observation)
    history = [first]
    boot = client.call_endpoint(
        "bootstrap",
        {
            "observation": nested(first),
            "capture_s": origin,
            "delay_ticks": d,
            "seed": seed,
        },
    )
    queue = CommandTimeline(boot["actions"])
    frames = []
    latencies = []
    actions_executed = []
    steps = calls = 0
    success = False
    counts = dict(boot["forward_counts"])
    while steps < args.max_steps:
        wanted, source, padded = initial_frame_index(steps, dv)
        obs = snapshot(observation)
        for key in obs:
            if key.startswith("video."):
                obs[key] = history[source][key].copy()
        committed = queue.reserve(steps, d)
        started = time.monotonic()
        result = client.call_endpoint(
            "plan",
            {
                "observation": {"state": nested(observation)["state"]},
                "request_tick": steps,
                "delay_ticks": d,
                "committed_actions": committed,
                "state_capture_s": origin + steps * period,
                "visual": {
                    "observation": nested(obs),
                    "capture_s": origin + wanted * period,
                    "source_tick": source,
                    "requested_tick": wanted,
                },
                "output_scope": "next_interval",
            },
        )
        if args.artificial_compute_wait:
            time.sleep(args.artificial_compute_wait)
        latencies.append(time.monotonic() - started)
        audit = result["audit"]
        calls += 1
        if audit["image_delay_ticks"] != dv:
            raise RuntimeError("Visual delay drifted")
        if audit["valid_start_tick"] - steps != d:
            raise RuntimeError("Action delay drifted")
        if audit["extra_buffer_shift"] != 0:
            raise RuntimeError("Fixed delay should not realign the buffer")
        if audit["forward_counts"]["vlm"] != 1:
            raise RuntimeError("Specified old image did not run through the VLM")
        publication = queue.publish(
            steps, audit["valid_start_tick"], result["actions"], next_tick=steps
        )
        if publication.expired or publication.protected:
            raise RuntimeError("Invalid controlled-time publication")
        for key in counts:
            counts[key] += audit["forward_counts"][key]
        trace.write(
            json.dumps(
                {
                    "kind": "request",
                    "request_tick": steps,
                    "requested_image_tick": wanted,
                    "source_image_tick": source,
                    "history_padding": padded,
                    "release_tick": steps + d,
                    "audit": audit,
                }
            )
            + "\n"
        )
        count = min(d, args.max_steps - steps)
        for _ in range(count):
            action, fallback, _age = queue.execute(steps)
            if fallback:
                raise RuntimeError(
                    "Fixed-delay algorithm protocol must not exhaust its queue"
                )
            actions_executed.append(action.copy())
            observation, _, done, truncated, info = checked_env_step(env, action)
            observation = snapshot(observation)
            steps += 1
            history.append(observation)
            if args.record_video:
                frames.append(observation["video.image"].copy())
            success = bool(info["success"])
            if success or done or truncated:
                break
        if success or done or truncated:
            break
    trajectory = np.stack(actions_executed)
    return {
        "success": success,
        "control_steps": steps,
        "policy_calls": calls,
        "forward_counts": counts,
        "action_trace_sha256": hashlib.sha256(trajectory.tobytes()).hexdigest(),
        "final_image_sha256": hashlib.sha256(
            observation["video.image"].tobytes()
        ).hexdigest(),
        "visual_delay_ticks": dv,
        "action_delay_ticks": d,
        "delay_contract_verified": True,
        "inference_wall_seconds": metric_summary(latencies),
        "bootstrap_seconds": boot["bootstrap_seconds"],
    }, frames


def save_summary(output, records, config):
    summary = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "episodes": len(records),
        "successes": sum(x["success"] for x in records),
        "success_rate": float(np.mean([x["success"] for x in records]))
        if records
        else None,
        "complete": len(records) == 10 * config["episodes_per_task"],
        "per_task": [],
    }
    if config["protocol"] == "deployment" and records:
        summary["timing"] = {
            "strict_valid_episodes": sum(x["control_rate_valid"] for x in records),
            "mean_rate_valid_episodes": sum(
                x["mean_control_rate_valid"] for x in records
            ),
            "deadline_valid_episodes": sum(
                x["control_deadline_valid"] for x in records
            ),
            "actual_hz": metric_summary(
                [
                    x["actual_control_hz"]
                    for x in records
                    if x["actual_control_hz"] is not None
                ]
            ),
            "control_deadline_misses": sum(
                x["control_deadline_misses_over5ms"] for x in records
            ),
            "action_request_deadline_misses": sum(
                x["action_request_deadline_misses"] for x in records
            ),
            "note": "Main success rate includes all trials; timing-invalid trials are never silently dropped.",
        }
    for task_id in range(10):
        rows = [x for x in records if x["task_id"] == task_id]
        if rows:
            summary["per_task"].append(
                {
                    "task_id": task_id,
                    "task": rows[0]["task"],
                    "episodes": len(rows),
                    "successes": sum(x["success"] for x in rows),
                }
            )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", choices=["algorithm", "deployment"], required=True
    )
    parser.add_argument("--variant", choices=["pir2", "flow"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5570)
    parser.add_argument("--slow-port", type=int, default=5572)
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=720)
    parser.add_argument("--control-hz", type=float, default=20)
    parser.add_argument("--visual-delay", type=int, default=3)
    parser.add_argument("--action-delay", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--artificial-compute-wait", type=float, default=0)
    args = parser.parse_args()
    if args.control_hz != 20:
        raise ValueError("This protocol must match LIBERO's physical20Hz control step")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".evaluation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    client = wait_server(args.port)
    identity = client.call_endpoint("identity", requires_input=False)
    assert identity["variant"] == args.variant
    config = {
        k: v for k, v in vars(args).items() if k not in ["output", "port", "slow_port"]
    }
    config["checkpoint"] = identity
    config["suite"] = "libero_spatial"
    config["initial_state_indices"] = list(range(args.episodes_per_task))
    config["version"] = 4
    config["reset_fix"] = "clear_property_samplers_on_model_reload"
    root = Path(__file__).resolve().parents[1]
    implementation_files = [
        "scripts/evaluate_libero_protocol.py",
        "scripts/libero_wallclock.py",
        "scripts/libero_protocol_scheduler.py",
        "scripts/libero_reproducible_env.py",
        "scripts/serve_libero_protocol.py",
        "scripts/libero_streaming_client.py",
        "scripts/libero_observations.py",
        "upstream/learning/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
        "upstream/learning/Isaac-GR00T/gr00t/policy/decoupled_policy.py",
    ]
    config["implementation_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in implementation_files
    }
    if args.protocol == "deployment":
        config.pop("visual_delay")
        config.pop("action_delay")
        slow = wait_server(args.slow_port)
        config["slow_checkpoint"] = slow.call_endpoint("identity", requires_input=False)
        if config["slow_checkpoint"]["role"] != "vlm":
            raise ValueError("The slow endpoint must be a VLM worker")
        del slow
        config["deployment_contract"] = {
            "control_deadline_slack_s": 0.005,
            "rate_mean_tolerance_fraction": 0.02,
            "no_future_observation_release": True,
            "cyclic_gc_disabled_during_control": True,
            "slow_vlm_warmup_calls_per_condition": 2,
            "strict_timing_validity": "mean frequency within2%, max lateness and adjacent interval error each<=5ms",
            "fallback": "zero Cartesian delta; retain last committed gripper",
            "flow_publication": "all native clean positions after committed prefix",
            "pir2_publication": "only next clean delay-sized segment",
            "action_delay_budget": "calibratedp95 plus5ms, nondecreasing within episode, maximum5ticks",
        }
    path = args.output / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("Protocol/checkpoint config changed on resume")
    path.write_text(json.dumps(config, indent=2) + "\n")
    records_path = args.output / "episodes.jsonl"
    records = (
        [json.loads(x) for x in records_path.read_text().splitlines()]
        if records_path.exists()
        else []
    )
    completed = {(x["task_id"], x["episode_id"]) for x in records}
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    for task_id in range(10):
        task = suite.get_task(task_id)
        states = suite.get_task_init_states(task_id)
        env = LiberoEnv(
            str(
                Path(get_libero_path("bddl_files"))
                / task.problem_folder
                / task.bddl_file
            ),
            task.language,
        )
        try:
            for episode_id in range(args.episodes_per_task):
                if (task_id, episode_id) in completed:
                    continue
                seed = args.seed + 1000 * task_id + episode_id
                np.random.seed(seed)
                env._env.seed(seed)
                env.reset()
                initial = np.asarray(states[episode_id])
                raw = env._env.set_init_state(initial)
                for _ in range(10):
                    raw, _, _, _ = env._env.step([0, 0, 0, 0, 0, 0, -1])
                obs = env._process_observation(raw)
                initial_fingerprints = env.initial_fingerprints(obs)
                args.record_video = episode_id == 0
                trace_path = (
                    args.output / f"task{task_id}-episode{episode_id}-trace.jsonl"
                )
                started = time.monotonic()
                with trace_path.open("w") as trace:
                    if args.protocol == "algorithm":
                        result, frames = algorithm_episode(
                            env, obs, client, args, seed, trace
                        )
                    else:
                        from libero_wallclock import deployment_episode

                        result, frames = deployment_episode(
                            env, obs, client, args, seed, trace
                        )
                result.update(initial_fingerprints)
                result.update(
                    task_id=task_id,
                    task=task.name,
                    episode_id=episode_id,
                    seed=seed,
                    initial_state_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                    wall_seconds=time.monotonic() - started,
                )
                if frames:
                    write_video(
                        args.output / f"task{task_id}-episode{episode_id}.mp4", frames
                    )
                records.append(result)
                with records_path.open("a") as f:
                    f.write(json.dumps(result) + "\n")
                save_summary(args.output, records, config)
                print("EPISODE " + json.dumps(result), flush=True)
        finally:
            env.close()
    save_summary(args.output, records, config)
    print("PROTOCOL_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
