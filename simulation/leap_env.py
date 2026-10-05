"""Paper-shaped Leap task built on the unmodified public Playground simulator."""

import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src.manipulation.leap_hand.reorient import (
    CubeReorient,
    default_config,
)

# Figure 6 (w,x,y,z): blue face up, four yaw angles.
_S = 2**-0.5
GOALS = jnp.array([[0, 1, 0, 0], [0, _S, _S, 0], [0, 0, 1, 0], [0, -_S, _S, 0]])


def orientation_error(quaternion):
    q = quaternion / jnp.maximum(jnp.linalg.norm(quaternion), 1e-8)
    goals = GOALS / jnp.linalg.norm(GOALS, axis=-1, keepdims=True)
    return 2 * jnp.arccos(jnp.clip(jnp.max(jnp.abs(goals @ q)), 0, 1))


class PaperLeap(CubeReorient):
    def __init__(self, ctrl_dt=0.02, impl="jax", batch_size=100):
        config = default_config()
        config.ctrl_dt = ctrl_dt
        config.episode_length = 600
        config.success_threshold = 0.2
        config.impl = impl
        config.naconmax = max(30 * batch_size, 1024)
        super().__init__(config)

    def reset(self, rng):
        initial_rng, goal_rng = jax.random.split(rng)
        state = super().reset(initial_rng)
        goal = GOALS[jax.random.randint(goal_rng, (), 0, 4)]
        data = mjx.forward(self.mjx_model, state.data.replace(mocap_quat=goal[None]))
        obs = self._get_obs(data, state.info)
        return state.replace(data=data, obs=obs)

    def _get_obs(self, data, info):
        obs = super()._get_obs(data, info)
        # Expert observes goal-relative orientation plus its previous action (57D).
        # Student receives only the paper's 41D, including noisy absolute orientation.
        relative_rows = obs["state"][35:41].reshape(2, 3)
        relative_rotation = jnp.concatenate(
            [jnp.cross(relative_rows[0], relative_rows[1])[None], relative_rows], axis=0
        )
        goal_rotation = math.quat_to_mat(self.get_cube_goal_orientation(data))
        absolute_rotation = relative_rotation @ goal_rotation
        obs["policy_state"] = jnp.concatenate(
            [obs["state"][:35], absolute_rotation.ravel()[3:]]
        )
        return obs

    def success(self, state):
        return (orientation_error(self.get_cube_orientation(state.data)) < 0.2) & (
            self.get_cube_position(state.data)[2] >= -0.05
        )


def make_advance(env, batch_size):
    """Advance one control tick inside scan, matching collection's stable path.

    With the pinned JAX/MJX stack, directly JIT-compiling the body materializes
    an unstable first physics step. A length-one scan passes the public-expert
    control (87/100 versus 85/100 for fused collection); direct JIT scores 0/100.
    Keep this wrapper covered by the cube-displacement GPU regression check.
    """
    step = jax.vmap(env.step)
    success = jax.vmap(env.success)

    def one(carry, command):
        current, alive, successes, lengths = carry
        proposed = step(current, command)
        reached = success(proposed) & alive
        lengths = lengths + alive.astype(jnp.int32)
        successes = successes | reached
        current = jax.tree.map(
            lambda new, old: jnp.where(
                alive.reshape((batch_size,) + (1,) * (new.ndim - 1)), new, old
            ),
            proposed,
            current,
        )
        alive = alive & ~reached & ~current.done.astype(bool)
        return (current, alive, successes, lengths), None

    @jax.jit
    def advance(state, action, alive, successes, lengths):
        return jax.lax.scan(one, (state, alive, successes, lengths), action[None])[0]

    return advance
