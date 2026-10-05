"""Evaluate the public GPU expert and optionally collect independent demonstrations."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from simulation.expert import load_expert
from simulation.leap_env import PaperLeap

ROOT = Path(__file__).resolve().parents[1]
EXPERT = ROOT / (
    "third_party/mujoco_playground/mujoco_playground/experimental/"
    "sim2sim/onnx/leap_reorient_policy.onnx"
)


def run(episodes, seed, steps, ctrl_dt, save_data, output):
    assert jax.default_backend() == "gpu", "JAX GPU is required"
    print("JAX devices:", jax.devices(), flush=True)
    env = PaperLeap(ctrl_dt=ctrl_dt, batch_size=episodes)
    policy = load_expert(EXPERT)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.vmap(env.step)
    success = jax.vmap(env.success)
    keys = jax.random.split(jax.random.PRNGKey(seed), episodes)
    started = time.monotonic()
    state = reset(keys)
    jax.block_until_ready(state)
    print("Reset complete; compiling rollout", flush=True)

    def rollout(initial):
        def advance(carry, _):
            current, alive, lengths, successes = carry
            observation = current.obs["policy_state"]
            action = policy(current.obs["state"])
            next_state = step(current, action)
            reached = success(next_state) & alive
            lengths = lengths + alive.astype(jnp.int32)
            successes = successes | reached
            next_state = jax.tree.map(
                lambda new, old: jnp.where(
                    alive.reshape((episodes,) + (1,) * (new.ndim - 1)), new, old
                ),
                next_state,
                current,
            )
            alive = alive & ~reached & ~next_state.done.astype(bool)
            record = (observation, action) if save_data else None
            return (next_state, alive, lengths, successes), record

        initial_carry = (
            initial,
            jnp.ones(episodes, bool),
            jnp.zeros(episodes, jnp.int32),
            jnp.zeros(episodes, bool),
        )
        return jax.lax.scan(advance, initial_carry, None, length=steps)

    final, records = jax.jit(rollout)(state)
    jax.block_until_ready(final)
    _, _, lengths, successes = final
    lengths, successes = np.asarray(lengths), np.asarray(successes)
    summary = {
        "kind": "public Playground expert, not piR2",
        "episodes": episodes,
        "seed": seed,
        "max_steps": steps,
        "control_hz": 1 / ctrl_dt,
        "success_count": int(successes.sum()),
        "success_rate": float(successes.mean()),
        "episode_successes": successes.tolist(),
        "episode_lengths": lengths.tolist(),
        "source": str(EXPERT.relative_to(ROOT)),
        "elapsed_wall_seconds": time.monotonic() - started,
        "difference_from_paper": "One public expert rather than four author-trained PPO experts",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    if save_data:
        obs, actions = (np.asarray(value).transpose(1, 0, 2) for value in records)
        assert obs.shape[-1] == 41 and actions.shape[-1] == 16
        assert np.isfinite(obs).all() and np.isfinite(actions).all()
        np.savez_compressed(
            output.with_suffix(".npz"),
            observations=obs,
            actions=actions,
            lengths=lengths,
            successes=successes,
        )
    print(
        json.dumps({k: v for k, v in summary.items() if not k.startswith("episode_")}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--ctrl-dt", type=float, default=0.02)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/simulation/expert-eval"
    )
    args = parser.parse_args()
    run(args.episodes, args.seed, args.steps, args.ctrl_dt, args.save_data, args.output)
