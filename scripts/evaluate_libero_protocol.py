"""Two separate clocks: controlled delays, or paced asynchronous deployment."""

import argparse
import fcntl
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from libero_protocol_scheduler import CommandTimeline, initial_frame_index

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def snapshot(observation):
    return {
        k: v if isinstance(v, str) else np.asarray(v).copy()
        for k, v in observation.items()
    }


def nested(observation):
    from libero_observations import batch_observation
    from libero_streaming_client import nested_observation

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
    from gr00t.policy.server_client import PolicyClient

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
        "physical_control_seconds": steps * period,
        "first_success_tick": steps - 1 if success else None,
        "measurement_mode": "success_terminated_closed_loop",
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
    task_ids = config.get("task_ids", list(range(10)))
    expected = {
        (task, episode)
        for task in task_ids
        for episode in range(config["episodes_per_task"])
    }
    completed = {(row["task_id"], row["episode_id"]) for row in records}
    summary = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "episodes": len(records),
        "successes": sum(x["success"] for x in records),
        "success_rate": float(np.mean([x["success"] for x in records]))
        if records
        else None,
        "expected_episodes": len(expected),
        "complete": completed == expected and len(records) == len(expected),
        "evaluation_scope": "full_spatial_suite"
        if sorted(task_ids) == list(range(10))
        else "selected_task_subset",
        "per_task": [],
    }
    if config["protocol"] == "deployment" and records:

        def timing_count(key):
            values = [row.get(key) for row in records]
            return sum(values) if all(value is not None for value in values) else None

        summary["timing"] = {
            "execution_clock": config.get("execution_clock", "paced"),
            "execution_coupling": config.get("execution_coupling", "asynchronous"),
            "fixed_rate_timing_applicable": config.get("execution_clock", "paced")
            == "paced"
            and config.get("execution_coupling", "asynchronous") == "asynchronous",
            "strict_valid_episodes": timing_count("control_rate_valid"),
            "mean_rate_valid_episodes": timing_count("mean_control_rate_valid"),
            "deadline_valid_episodes": timing_count("control_deadline_valid"),
            "actual_hz": metric_summary(
                [
                    x["actual_control_hz"]
                    for x in records
                    if x["actual_control_hz"] is not None
                ]
            ),
            "control_deadline_misses": timing_count("control_deadline_misses_over5ms"),
            "action_request_deadline_misses": timing_count(
                "action_request_deadline_misses"
            ),
            "real_time_factor": metric_summary(
                [
                    row["real_time_factor"]
                    for row in records
                    if row.get("real_time_factor") is not None
                ]
            ),
            "note": "All trials remain included. Ready or synchronous execution has no fixed global wall-rate deadline; null validity is not a timing pass or failure.",
        }
    for task_id in task_ids:
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


def build_parser():
    from coexecution.libero.features import add_feature_arguments

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", choices=["algorithm", "deployment"], required=True
    )
    parser.add_argument("--variant", choices=["pir2", "flow"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5570)
    parser.add_argument("--slow-port", type=int, default=5572)
    parser.add_argument("--server-layout", choices=("split", "paired"), default="split")
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument(
        "--task-ids", nargs="+", type=int, choices=range(10), default=list(range(10))
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--record-replay",
        action="store_true",
        help="Deployment debug: owned robot states and both cameras; WebM encoding after control",
    )
    parser.add_argument("--replay-max-frames", type=int, default=1024)
    parser.add_argument("--slow-warmup-calls", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=720)
    parser.add_argument("--control-hz", type=float, default=20)
    parser.add_argument(
        "--execution-coupling",
        choices=("asynchronous", "synchronous", "one_step_async", "chunked_async"),
        default="asynchronous",
        help="Flow: synchronous alternates inference/execution; chunked_async prefetches D steps before a K-step boundary; one_step_async preserves K1/D1",
    )
    parser.add_argument(
        "--synchronous-chunk-steps",
        "--execution-prefix-steps",
        dest="synchronous_chunk_steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--async-overlap-steps",
        type=int,
        default=None,
        help="Chunked async prefetch depth D, 1<=D<=K; synchronous accepts D0",
    )
    parser.add_argument(
        "--synchronous-warmup-timeout-seconds",
        type=int,
        default=900,
        help="Synchronous cold-JIT warmup socket timeout; scored RPC timeout is unchanged",
    )
    parser.add_argument(
        "--execution-clock",
        choices=("paced", "ready"),
        default="paced",
        help="Deployment Flow: ready accepts plans between atomic physics steps without wall pacing",
    )
    parser.add_argument("--visual-delay", type=int, default=3)
    parser.add_argument("--action-delay", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--artificial-compute-wait", type=float, default=0)
    add_feature_arguments(parser, rpc=True)
    parser.add_argument(
        "--action-min-service-ms",
        type=float,
        default=0,
        help="Deployment: hold real DiT result until submission plus this duration",
    )
    parser.add_argument(
        "--vision-min-service-ms",
        type=float,
        default=0,
        help="Deployment: hold real VLM result until worker start plus this duration",
    )
    parser.add_argument(
        "--vision-max-hz",
        type=float,
        default=0,
        help="Deployment: cap scored VLM admissions; zero keeps unlimited admission",
    )
    parser.add_argument(
        "--deployment-scheduling-policy",
        choices=("conservative", "responsive"),
        default="conservative",
        help="Deployment piR2: opt into bounded delay recovery and prompt result publication",
    )
    parser.add_argument(
        "--deployment-fixed-delay",
        type=int,
        choices=range(1, 6),
        help="Deployment: fix bootstrap and control d; disable adaptive increases",
    )
    return parser


def validate_deployment_controls(args):
    from libero_wallclock import validate_service_floor

    from coexecution.libero.features import validate_feature_options
    from coexecution.libero.scheduling import (
        validate_deployment_scheduling_policy,
        validate_execution_clock,
        validate_vision_max_hz,
    )
    from coexecution.libero.synchronous import validate_synchronous_controls

    validate_deployment_scheduling_policy(
        getattr(args, "deployment_scheduling_policy", "conservative"),
        protocol=args.protocol,
        variant=args.variant,
        execution_coupling=getattr(args, "execution_coupling", "asynchronous"),
        execution_clock=getattr(args, "execution_clock", "paced"),
        fixed_delay=args.deployment_fixed_delay,
    )
    validate_synchronous_controls(args)
    validate_execution_clock(
        getattr(args, "execution_clock", "paced"),
        protocol=args.protocol,
        variant=args.variant,
        fixed_delay=args.deployment_fixed_delay,
    )

    if getattr(args, "record_replay", False) and args.protocol != "deployment":
        raise ValueError("Replay recording requires the deployment protocol")
    if getattr(args, "replay_max_frames", 1024) < 1:
        raise ValueError("Replay frame bound must be positive")
    validate_service_floor(args.action_min_service_ms)
    validate_service_floor(args.vision_min_service_ms)
    validate_vision_max_hz(args.vision_max_hz)
    validate_feature_options(args.feature_transport, args.feature_rpc)
    if args.feature_transport == "cuda_local" and (
        args.server_layout != "paired" or args.variant != "flow"
    ):
        raise ValueError("cuda_local requires paired Flow workers")
    if args.protocol != "deployment" and (
        args.action_min_service_ms
        or args.vision_min_service_ms
        or args.vision_max_hz
        or args.deployment_fixed_delay is not None
        or args.feature_transport != "numpy_fp32"
        or args.feature_rpc != "separate"
    ):
        raise ValueError(
            "Service floors, vision admission limits, feature controls and fixed delay require the deployment protocol"
        )


def save_run_configuration(output, config):
    """Bind resumes to stable provenance and retain each server invocation separately."""
    identities = {
        key: config[key] for key in ("checkpoint", "slow_checkpoint") if key in config
    }
    stable_config = {
        **config,
        **{
            key: {
                name: value
                for name, value in identity.items()
                if name
                not in {
                    "pid",
                    "cuda_stream",
                    "server_instance",
                    "policy_instance",
                    "feature_exchange",
                }
            }
            | (
                {
                    "feature_exchange": {
                        "max_leases": identity["feature_exchange"]["max_leases"]
                    }
                }
                if "feature_exchange" in identity
                else {}
            )
            for key, identity in identities.items()
        },
    }
    path = output / "config.json"
    if path.exists() and json.loads(path.read_text()) != stable_config:
        raise RuntimeError("Protocol/checkpoint config changed on resume")
    path.write_text(json.dumps(stable_config, indent=2) + "\n")
    invocation = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "identities": identities,
    }
    with (output / "runtime-identities.jsonl").open("a") as stream:
        stream.write(json.dumps(invocation) + "\n")
    return stable_config


def main():
    args = build_parser().parse_args()
    validate_deployment_controls(args)
    if args.control_hz != 20:
        raise ValueError("This protocol must match LIBERO's physical20Hz control step")
    if len(set(args.task_ids)) != len(args.task_ids):
        raise ValueError("Task IDs must be unique")
    if args.episodes_per_task < 1 or args.max_steps < 1:
        raise ValueError("At least one episode and control step are required")
    if args.slow_warmup_calls < 3:
        raise ValueError(
            "At least three VLM warmup calls are required (first two excluded)"
        )
    from libero.libero import benchmark, get_libero_path
    from libero_observations import write_video
    from libero_reproducible_env import LiberoEnv

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
    config["version"] = 10
    config["measurement_mode"] = "success_terminated_closed_loop"
    config["input_dependencies"] = {
        "camera": "physical observation released at each control tick; latest waiting frame replaces older waiting frames",
        "vlm": "one in-flight visual request; last completed real VLM feature is eligible when capture/source tick do not exceed state",
        "action": "current robot state, selected completed VLM feature, immutable committed action prefix",
        "bootstrap": "fresh physical image and state captured together before timed control",
        "termination": "first success, environment termination, or max steps; actual exposure reported per episode",
    }
    if args.protocol == "algorithm":
        config["input_dependencies"].update(
            camera="retained observation history at requested fixed visual delay; initial history padding is explicit",
            vlm="one real VLM forward for each requested historical image before planning",
            action="current robot state, fixed-delay historical image feature, immutable committed prefix; compute waits do not advance physics",
            bootstrap="initial image and state before controlled simulated time",
        )
    config["reset_fix"] = "clear_property_samplers_on_model_reload"
    root = Path(__file__).resolve().parents[1]
    implementation_files = [
        "scripts/evaluate_libero_protocol.py",
        "scripts/libero_wallclock.py",
        "scripts/libero_protocol_scheduler.py",
        "scripts/libero_queue_trace.py",
        "scripts/libero_reproducible_env.py",
        "scripts/libero_observations.py",
        "coexecution/replay/recording.py",
        "coexecution/replay/schema.py",
        "scripts/serve_libero_protocol.py",
        "scripts/libero_inference_backend.py",
        "scripts/libero_model_loading.py",
        "coexecution/libero/loading.py",
        "coexecution/libero/backend.py",
        "coexecution/libero/service.py",
        "coexecution/libero/scheduling.py",
        "coexecution/libero/features.py",
        "coexecution/libero/resident.py",
        "coexecution/libero/synchronous.py",
        "coexecution/libero/one_step_async.py",
        "coexecution/optimization/execution/submission.py",
        "scripts/libero_streaming_client.py",
        "scripts/libero_observations.py",
        "upstream/learning/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
        "upstream/learning/Isaac-GR00T/gr00t/policy/decoupled_policy.py",
    ]
    config["implementation_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in implementation_files
    }
    if args.protocol == "deployment" and args.execution_coupling in (
        "synchronous",
        "one_step_async",
        "chunked_async",
    ):
        from coexecution.libero.synchronous import validate_synchronous_identity

        validate_synchronous_identity(identity, args.synchronous_chunk_steps)
        config.pop("visual_delay")
        config.pop("action_delay")
        config["input_dependencies"].update(
            camera="one current physical observation before each inference cycle; simulator does not advance during inference",
            vlm="fresh real VLM on the same single policy endpoint before every Flow4 plan",
            action="native decoded clean Flow4 prefix; no inference during execution and no committed stale prefix",
            bootstrap="task-specific full-plan warmup before one episode RNG reset; no scored bootstrap",
        )
        config.update(
            deployment_contract={
                "execution_coupling": "synchronous",
                "execution_clock": args.execution_clock,
                "physics_step_seconds": 0.05,
                "nominal_physics_hz": 20,
                "target_wall_hz": None,
                "phase_target_hz": 20 if args.execution_clock == "paced" else None,
                "fixed_rate_timing_applicable": False,
                "strict_timing_validity": None,
                "execution_rule": "fresh VLM plus complete Flow4, then up to synchronous_chunk_steps clean commands; stop on first success/termination",
                "pacing": "paced waits only within each execution phase, including final command dwell; ready adds no wait",
                "clocks": "wall time continues while inference freezes physics; each env.step advances exactly 0.05 simulated seconds",
                "timing_scope": "instrumented closed-loop debug; real server stage and controller timestamps, no synthetic GPU timeline",
            }
        )
        if args.execution_coupling in ("one_step_async", "chunked_async"):
            chunked = args.execution_coupling == "chunked_async"
            prefix = args.synchronous_chunk_steps
            overlap = args.async_overlap_steps if chunked else 1
            config["execution_prefix_steps"] = prefix
            config["async_overlap_steps"] = overlap
            config["input_dependencies"].update(
                camera="owned image and state snapshot before executing current action; one prefetch only",
                vlm="fresh real VLM then complete Flow4 on one policy endpoint for every request",
                action="first decoded command only; bootstrap source0/target0, then source k/target k+1; no state prediction or compensating action suffix",
                bootstrap="task-specific full warmup then one RNG reset; scored first policy request must complete before action0",
            )
            config["deployment_contract"].update(
                execution_coupling=args.execution_coupling,
                execution_prefix_steps=prefix,
                async_overlap_steps=overlap,
                execution_rule="submit next full policy from pre-step observation, execute one current command, wait explicitly if next result is not ready; drain unused terminal work",
                pacing="ready only; no additional wall pacing and no buffered action while waiting",
                clocks="each env.step advances 0.05 simulated seconds; full policy inference can overlap actual step wall time; no 50ms wall-overlap assumption",
                observation_lag="0 physics ticks for bootstrap; exactly1 physics tick thereafter; wall age measured from capture to application separately",
                worker_ownership="dedicated worker creates, uses, and destroys its own PolicyClient; simulator owner alone observes and steps",
                accounting="controlled window ends after final action; async_work_end retains unused speculative RPC and drain tail separately",
            )
            if chunked:
                config["input_dependencies"].update(
                    camera="owned image/state at bootstrap and target-D before the current atomic step; reuse post-step observation within a prefix",
                    action="commit native offsets0..K-1; next target T uses source T-D; final cap accepts min(K,max_steps-T); stop immediately on success/termination",
                )
                config["deployment_contract"].update(
                    protocol_version=2,
                    execution_rule="submit next full policy D steps before boundary, finish current committed prefix even if result is early, adopt or wait at boundary; drain unused terminal work",
                    observation_lag="bootstrap offset j: j; later offset j: D+j physical ticks; capture-to-apply wall AoI separate",
                    accounting="accepted prefix, executed offset count, unexecuted accepted steps, discarded native suffix and all unused native actions are distinct",
                )
    elif args.protocol == "deployment":
        from coexecution.libero.features import (
            validate_feature_identity,
            validate_worker_pair,
        )

        config.pop("visual_delay")
        config.pop("action_delay")
        slow = wait_server(args.slow_port)
        config["slow_checkpoint"] = slow.call_endpoint("identity", requires_input=False)
        for worker_identity in (identity, config["slow_checkpoint"]):
            validate_feature_identity(worker_identity, args.feature_transport)
        validate_worker_pair(
            identity,
            config["slow_checkpoint"],
            args.server_layout,
            args.feature_transport,
        )
        if args.feature_rpc == "combined" and "combined" not in identity.get(
            "feature_rpc_modes", []
        ):
            raise ValueError(
                "Action worker does not declare combined feature RPC support"
            )
        if config["slow_checkpoint"]["role"] != "vlm":
            raise ValueError("The slow endpoint must be a VLM worker")
        del slow
        config["deployment_contract"] = {
            "execution_clock": args.execution_clock,
            "physics_step_seconds": 0.05,
            "nominal_physics_hz": 20,
            "target_wall_hz": 20 if args.execution_clock == "paced" else None,
            "control_deadline_slack_s": 0.005,
            "rate_mean_tolerance_fraction": 0.02,
            "no_future_observation_release": True,
            "cyclic_gc_disabled_during_control": True,
            "slow_vlm_warmup_calls_per_condition": args.slow_warmup_calls,
            "slow_vlm_calibration_excluded_initial_calls": 2,
            "episode_vision_preparation": {
                "calls_per_episode": 1,
                "scope": "real S2 VLM before fresh S1 bootstrap capture and scored control, for every layout and transport",
                "publication": "return_features=False; preparation outputs are never published or installed in S1",
                "evidence": "episode vision_preparation contains original source/clock times, physical VLM count and graph deltas; scored cold captures remain separately reported",
            },
            "strict_timing_validity": "mean frequency within2%, max lateness and adjacent interval error each<=5ms",
            "fallback": "zero Cartesian delta; retain last committed gripper",
            "flow_publication": "all native clean positions after committed prefix",
            "pir2_publication": "only next clean delay-sized segment",
            "deployment_scheduling_policy": args.deployment_scheduling_policy,
            "controller_result_ingestion": "controller-only during pacing after atomic step; callback signals only; no early sensor release or S1 submission"
            if args.deployment_scheduling_policy == "responsive"
            else "at physical tick boundary",
            "action_delay_budget": (
                "fixed bootstrap and control delay; misses and expired/protected slots still recorded"
                if args.deployment_fixed_delay is not None
                else "solo calibrated initial d; completed RPC rolling8 P95+5ms; lower one tick after3 full-window supports; raise on deadline miss including dispatch lateness; clamp1..5"
                if args.deployment_scheduling_policy == "responsive"
                else "calibratedp95 plus5ms, nondecreasing within episode, maximum5ticks"
            ),
            "service_floor_contract": {
                "scope": "real neural RPC then client-side sleep; no GPU work added and no compute acceleration claimed",
                "action_anchor": "submission includes executor queue, feature installation, and plan RPC; result future withheld until release",
                "vision_anchor": "worker RPC start; feature publication and next VLM admission withheld until release",
                "target_overruns": "real completion beyond target retained; raw and visible times both recorded",
                "calibration": "same per-role floors as scored workers; bootstrap is outside scored service timing",
            },
            "vision_admission_contract": {
                "max_hz": args.vision_max_hz,
                "scope": "scored asynchronous vision worker only; bootstrap and solo warmup remain uncapped",
                "anchor": "actual monotonic admission under mailbox lock, selecting the newest pending frame; distinct from RPC started_s",
                "spacing": "zero disables the limit; first admission immediate, subsequent admissions at least 1/max_hz after the previous actual admission; no catch-up",
                "camera": "physical 20Hz offers continue, replacing the latest waiting frame until admission",
                "publication": "admission waits never hold completed feature publication; service floors remain independent",
                "stop": "no new admission after stop observed under the same lock; already admitted RPCs may start or complete afterward; control cutoff stays unchanged",
            },
            "feature_transport_contract": {
                "encoding": args.feature_transport,
                "server_layout": args.server_layout,
                "rpc": args.feature_rpc,
                "timing_version": 2,
                "selection": "unchanged completed-feature selection and sequence deduplication; combined replies advance client installation only on success",
                "plan_rpc_started_s": "actual client plan network call entry; combined calls include feature installation",
                "server_plan_started_s": "server plan entry after optional feature installation and stream readiness; preferred cache-age boundary, not the first neural kernel",
                "server_seconds": "server plan wall time after installation; server_request_seconds also includes combined installation",
                "whole_call": "submission through completion includes executor queue, installation, network and planning in both modes",
                "clock": "loopback workers and client share the host monotonic clock",
                "representation": "numpy_bf16 preserves BF16 bits; cuda_local uses bounded same-process leases, producer completion, consumer event wait and stream ownership; independent of model precision",
            },
        }
        if args.execution_clock == "ready":
            config["input_dependencies"]["camera"] = (
                "actual post-step observation captured when the atomic env.step completes; released at the next controller boundary with no wall grid"
            )
            config["deployment_contract"].update(
                control_deadline_slack_s=None,
                rate_mean_tolerance_fraction=None,
                strict_timing_validity=None,
                execution_rule="accept completed GR00T plans at the first controller boundary after an atomic env.step; execute the next legal command immediately without wall pacing or final padding",
                action_delay_budget="one-step committed logical prefix; RPC calibration retained as evidence only; no wall-lateness adaptation",
                model_semantics="unchanged GR00T Flow clean action chunk; no StreamingVLA AFM, AEO or partially denoised action execution",
            )
            config["deployment_contract"]["vision_admission_contract"]["camera"] = (
                "one fresh observation offered after each atomic 0.05-s physics step; wall rate follows actual execution, waiting frames remain replaceable"
            )
    config = save_run_configuration(args.output, config)
    records_path = args.output / "episodes.jsonl"
    records = (
        [json.loads(x) for x in records_path.read_text().splitlines()]
        if records_path.exists()
        else []
    )
    completed = {(x["task_id"], x["episode_id"]) for x in records}
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    for task_id in args.task_ids:
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
                args.record_video = (
                    episode_id == 0 and not args.no_video and not args.record_replay
                )
                trace_path = (
                    args.output / f"task{task_id}-episode{episode_id}-trace.jsonl"
                )
                started = time.monotonic()
                with trace_path.open("w") as trace:
                    if args.protocol == "algorithm":
                        result, frames = algorithm_episode(
                            env, obs, client, args, seed, trace
                        )
                    elif args.execution_coupling in ("one_step_async", "chunked_async"):
                        from coexecution.libero.one_step_async import (
                            one_step_async_episode,
                        )

                        result, frames = one_step_async_episode(
                            env, obs, args, seed, trace
                        )
                    elif args.execution_coupling == "synchronous":
                        from coexecution.libero.synchronous import synchronous_episode

                        result, frames = synchronous_episode(
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
                if args.record_replay:
                    result["replay_recording"] = args.replay_recorder.finish(
                        args.output, f"task{task_id}-episode{episode_id}"
                    )
                    args.replay_recorder = None
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
