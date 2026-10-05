"""Control-tick piR2 client with an optional independent asynchronous VLM worker."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from gr00t.policy.server_client import PolicyClient

ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def nested_observation(flat):
    return {
        "video": {k[6:]: v for k, v in flat.items() if k.startswith("video.")},
        "state": {k[6:]: v for k, v in flat.items() if k.startswith("state.")},
        "language": {
            k: [[v[0]]]
            for k, v in flat.items()
            if not k.startswith(("video.", "state."))
        },
    }


def projected_prefix(actions, start, length):
    raw = np.concatenate(
        [actions[k][0, start : start + length] for k in ACTION_KEYS], axis=-1
    ).copy()
    raw[:, :6] = np.clip(raw[:, :6], -1, 1)
    raw[:, 6] = (raw[:, 6] >= 0.5).astype(np.float32)
    return raw


class StreamingClient:
    def __init__(self, client, mode, slide_steps, slow_port=5561):
        self.client = client
        self.mode = mode
        self.d = slide_steps
        self.slow_port = slow_port
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.local = threading.local()
        self.pending = None
        self.seeded = False
        self.frame_hashes = {}
        self.committed = None
        self.vlm_counts = {
            "bootstrap": 0,
            "synchronous": 0,
            "submitted": 0,
            "completed": 0,
            "published": 0,
        }

    def _slow_update(self, obs, capture):
        if not hasattr(self.local, "client"):
            self.local.client = PolicyClient(
                host="127.0.0.1", port=self.slow_port, timeout_ms=120000
            )
        return self.local.client.call_endpoint(
            "update_vlm_cache",
            {"observation": obs, "return_features": True, "t_image_capture": capture},
        )

    def reset(self, options=None):
        # Drain prior-episode work; its returned features are deliberately discarded.
        self.finish_episode()
        self.pending = None
        self.seeded = False
        self.frame_hashes = {}
        self.committed = None
        self.vlm_counts = {
            "bootstrap": 0,
            "synchronous": 0,
            "submitted": 0,
            "completed": 0,
            "published": 0,
        }
        return self.client.reset(options)

    def get_action(self, flat, options=None):
        options = options or {}
        started = time.monotonic()
        obs = nested_observation(flat)
        # Both timestamps use simulator control time, not mixed wall/simulation clocks.
        capture = 1.0 + options["control_step"] / 20.0
        counts = {"vlm": 0, "dit": 0}
        if not self.seeded:
            seed, info = self.client.call_endpoint(
                "seed_streaming_from_obs",
                {
                    "observation": obs,
                    "num_inference_timesteps": 4,
                    "t_image_capture": capture,
                    "slide_steps": self.d,
                },
            )
            self.committed = projected_prefix(seed, 0, self.d)
            self.frame_hashes = info["audit"]["frame_sha256"]
            for k in counts:
                counts[k] += info["audit"]["forward_counts"][k]
            self.seeded = True
            self.vlm_counts["bootstrap"] += info["audit"]["forward_counts"]["vlm"]
        elif self.mode == "pir2-sync":
            update = self.client.call_endpoint(
                "update_vlm_cache", {"observation": obs, "t_image_capture": capture}
            )
            self.frame_hashes = update["audit"]["frame_sha256"]
            counts["vlm"] += update["audit"]["forward_counts"]["vlm"]
            self.vlm_counts["synchronous"] += update["audit"]["forward_counts"]["vlm"]
        else:
            if self.pending is not None and self.pending.done():
                update = self.pending.result()
                self.pending = None
                self.vlm_counts["completed"] += update["audit"]["forward_counts"]["vlm"]
                self.vlm_counts["published"] += 1
                self.client.call_endpoint(
                    "install_vlm_cache",
                    {
                        "vl_embeds": update["vl_embeds"],
                        "t_image_capture": update["t_image_capture"],
                    },
                )
                self.frame_hashes = update["audit"]["frame_sha256"]
            if self.pending is None:
                self.pending = self.pool.submit(self._slow_update, obs, capture)
                self.vlm_counts["submitted"] += 1
        actions, info = self.client.call_endpoint(
            "get_action_chunk_cached",
            {
                "observation": {"state": obs["state"]},
                "options": {"slide_steps": self.d, "num_inference_steps_per_call": 1},
                "t_state_capture": capture,
                "period_ms": 50.0,
                "committed_actions": self.committed,
            },
        )
        for k in counts:
            counts[k] += info["audit"]["forward_counts"][k]
        # Front d is the already-committed segment; the next segment becomes the
        # prefix for the following call. Never execute the seed separately.
        self.committed = projected_prefix(actions, self.d, self.d)
        front = projected_prefix(actions, 0, self.d)
        for idx, k in enumerate(ACTION_KEYS):
            actions[k][0, : self.d, 0] = front[:, idx]
        return {"action." + k: v for k, v in actions.items()}, {
            "audit": {
                "forward_counts": counts,
                "frame_sha256": self.frame_hashes,
                "inference_seconds": time.monotonic() - started,
                "action_head_seconds": info["audit"]["inference_seconds"],
                "image_delay_ticks": info["image_delay_ticks"],
                "image_delay_ms": info["image_delay_ms"],
                "cache_id_used": info["cache_id_used"],
                "mode": self.mode,
            }
        }

    def finish_episode(self):
        if self.pending is not None:
            update = self.pending.result()
            self.pending = None
            self.vlm_counts["completed"] += update["audit"]["forward_counts"]["vlm"]
        return {
            **self.vlm_counts,
            "total_completed_vlm": self.vlm_counts["bootstrap"]
            + self.vlm_counts["synchronous"]
            + self.vlm_counts["completed"],
        }

    def close(self):
        self.finish_episode()
        self.pool.shutdown(wait=True)
