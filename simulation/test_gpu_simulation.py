"""GPU checks that protect expert conversion and success-metric semantics."""

import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
import torch

from simulation.expert import load_expert
from simulation.leap_env import GOALS, PaperLeap, make_advance, orientation_error

EXPERT = Path(__file__).resolve().parents[1] / (
    "third_party/mujoco_playground/mujoco_playground/experimental/"
    "sim2sim/onnx/leap_reorient_policy.onnx"
)


class TestSimulationGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert jax.default_backend() == "gpu"
        assert torch.cuda.is_available()

    def test_onnx_conversion_matches_cuda_reference(self):
        ort.preload_dlls(directory="")
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = ort.InferenceSession(
            str(EXPERT), options, providers=["CUDAExecutionProvider"]
        )
        assert session.get_providers()[0] == "CUDAExecutionProvider"
        expert = jax.jit(load_expert(EXPERT))
        for seed in range(3):
            obs = np.random.default_rng(seed).normal(size=(1, 57)).astype(np.float32)
            expected = session.run(None, {"obs": obs})[0]
            actual = np.asarray(expert(obs))
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)

    def test_external_physics_does_not_teleport_cube_on_first_step(self):
        env = PaperLeap(batch_size=100)
        state = jax.jit(jax.vmap(env.reset))(
            jax.random.split(jax.random.PRNGKey(100000), 100)
        )
        expert = jax.jit(load_expert(EXPERT))
        advance = make_advance(env, 100)
        alive = jnp.ones(100, bool)
        successes = jnp.zeros(100, bool)
        lengths = jnp.zeros(100, jnp.int32)
        initial_pos = jax.vmap(env.get_cube_position)(state.data)
        state, alive, successes, lengths = advance(
            state, expert(state.obs["state"]), alive, successes, lengths
        )
        final_pos = jax.vmap(env.get_cube_position)(state.data)
        self.assertEqual(int(alive.sum() + successes.sum()), 100)
        self.assertLess(
            float(jnp.linalg.norm(final_pos - initial_pos, axis=-1).max()), 0.1
        )

    def test_success_metric_uses_all_four_goals_and_quaternion_sign(self):
        errors = np.asarray(jax.jit(jax.vmap(orientation_error))(GOALS))
        negated = np.asarray(jax.jit(jax.vmap(orientation_error))(-GOALS))
        np.testing.assert_allclose(errors, 0, atol=1e-5)
        np.testing.assert_allclose(negated, 0, atol=1e-5)
        self.assertGreater(float(orientation_error(np.array([1, 0, 0, 0]))), 3.0)


if __name__ == "__main__":
    unittest.main()
