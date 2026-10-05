"""GPU checks for the independently implemented flow policy and rolling buffer."""

import unittest

import torch

from simulation.model import FlowUNet, clamp_action, schedule, stream_step


class TestPolicyGPU(unittest.TestCase):
    def test_inpaint_prefix_matches_saturated_physical_commands(self):
        normalized = torch.tensor([[-4.0, 3.0, 0.1]], device="cuda")
        offset = torch.tensor([0.5, -0.5, 0.0], device="cuda")
        scale = torch.tensor([2.0, 0.5, 0.2], device="cuda")
        clamped = clamp_action(normalized, offset, scale)
        torch.testing.assert_close(
            clamped * scale + offset,
            torch.tensor([[-1.0, 1.0, 0.02]], device="cuda"),
        )

    def test_oracle_flow_emits_clean_actions_and_preserves_inflight_prefix(self):
        assert torch.cuda.is_available()
        for delay in [1, 2, 3, 5]:
            times = schedule(16, delay, device="cuda").expand(3, -1)
            noise, clean = [torch.randn(3, 16, 16, device="cuda") for _ in range(2)]
            buffer = noise * (1 - times[..., None]) + clean * times[..., None]
            calls = []

            def oracle(_buffer, _times, _obs, calls=calls, clean=clean, noise=noise):
                calls.append(1)
                return clean - noise

            pred, shifted, next_times = stream_step(oracle, buffer, times, None, delay)
            self.assertEqual(len(calls), 1)
            torch.testing.assert_close(pred[:, :delay], buffer[:, :delay])
            torch.testing.assert_close(
                pred[:, delay : 2 * delay], clean[:, delay : 2 * delay]
            )
            torch.testing.assert_close(shifted[:, :-delay], pred[:, delay:])
            torch.testing.assert_close(next_times, times)

    def test_unet_trains_with_position_times_and_observation_history(self):
        model = FlowUNet(dims=(32, 64, 128)).cuda()
        x = torch.randn(4, 16, 16, device="cuda")
        times = torch.rand(4, 16, device="cuda")
        obs = torch.randn(4, 2, 41, device="cuda", requires_grad=True)
        y = model(x, times, obs)
        self.assertEqual(y.shape, x.shape)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(y).all())
        self.assertGreater(obs.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
