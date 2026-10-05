"""Real-GPU timing invariance and rolling-buffer recovery checks; not a success-rate benchmark."""

import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from evaluate_libero_protocol import algorithm_episode, nested, wait_server
from libero.libero import benchmark, get_libero_path
from libero_reproducible_env import LiberoEnv

ROOT = Path(__file__).resolve().parents[1]


def reset(env, state, seed):
    np.random.seed(seed)
    env._env.seed(seed)
    env.reset()
    raw = env._env.set_init_state(np.asarray(state))
    for _ in range(10):
        raw, _, _, _ = env._env.step([0, 0, 0, 0, 0, 0, -1])
    return env._process_observation(raw)


def main():
    client = wait_server(5570)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    args = SimpleNamespace(
        control_hz=20,
        action_delay=1,
        visual_delay=3,
        max_steps=20,
        record_video=False,
        artificial_compute_wait=0,
    )
    records = []
    for task_id in range(10):
        task = suite.get_task(task_id)
        state = suite.get_task_init_states(task_id)[0]
        env = LiberoEnv(
            str(
                Path(get_libero_path("bddl_files"))
                / task.problem_folder
                / task.bddl_file
            ),
            task.language,
        )
        try:
            outputs = []
            for wait in [0.0, 0.1]:
                obs = reset(env, state, 1000 + 1000 * task_id)
                args.artificial_compute_wait = wait
                result, _ = algorithm_episode(
                    env, obs, client, args, 1000 + 1000 * task_id, io.StringIO()
                )
                outputs.append(result)
            equal = (
                outputs[0]["action_trace_sha256"] == outputs[1]["action_trace_sha256"]
                and outputs[0]["final_image_sha256"] == outputs[1]["final_image_sha256"]
            )
            records.append(
                {
                    "task_id": task_id,
                    "control_ticks": 20,
                    "extra_wall_wait_per_call_seconds": 0.1,
                    "action_and_final_image_equal": equal,
                }
            )
            if not equal:
                raise RuntimeError(f"Timing invariance failed for task{task_id}")
            if task_id == 0:
                obs = reset(env, state, 1000)
                t = time.monotonic()
                boot = client.call_endpoint(
                    "bootstrap",
                    {
                        "observation": nested(obs),
                        "capture_s": t,
                        "delay_ticks": 1,
                        "seed": 1000,
                    },
                )
                # Deliberately expire a complete buffer, without scoring a task.
                response = client.call_endpoint(
                    "plan",
                    {
                        "observation": {"state": nested(obs)["state"]},
                        "request_tick": 44,
                        "delay_ticks": 1,
                        "committed_actions": boot["actions"][:1],
                        "state_capture_s": t,
                        "output_scope": "native",
                    },
                )
                audit = response["audit"]
                assert (
                    audit["buffer_restarted"]
                    and audit["forward_counts"]["dit"] == 5
                    and audit["buffer_origin_after"] == 45
                )
                recovery = {
                    "restarted": True,
                    "dit_evaluations": 5,
                    "next_buffer_origin": 45,
                }
        finally:
            env.close()
        print("GPU_CONTRACT", records[-1], flush=True)
    output = {
        "scope": "real GPU contract validation only, not task success rate",
        "fixed_delay_invariance": records,
        "full_horizon_recovery": recovery,
        "passed": True,
    }
    (ROOT / "artifacts/libero-protocols/gpu-contracts.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print("GPU_PROTOCOL_CONTRACTS_PASSED", flush=True)


if __name__ == "__main__":
    main()
