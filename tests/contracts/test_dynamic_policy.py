"""Default GR00T bridge contracts; host doubles never construct a model."""

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest


class Client:
    def __init__(self, identity=None, mutate=None):
        self.owner = threading.get_ident()
        self.calls = []
        self.closed = False
        self.mutate = mutate
        self.identity = {
            "variant": "flow",
            "role": "action",
            "server_layout": "split",
            "feature_transport": "numpy_fp32",
            "execution_endpoints": ["synchronous_plan"],
            "predicted_horizon": 40,
            "decoded_horizon": 16,
        }
        self.identity.update(identity or {})
        self.result = None

    def call_endpoint(self, endpoint, data=None, requires_input=True):
        assert threading.get_ident() == self.owner
        self.calls.append((endpoint, copy.deepcopy(data)))
        if endpoint == "identity":
            assert not requires_input
            return self.identity
        if endpoint == "reset":
            return {"reset": True}
        assert endpoint == "synchronous_plan"
        actions = np.zeros((16, 7), dtype=np.float32)
        actions[:, 0] = np.arange(16) / 20
        actions[:, 6] = np.arange(16) % 2
        self.result = {
            "actions": actions,
            "audit": {
                "source_tick": data["source_tick"],
                "capture_s": data["capture_s"],
                "decoded_horizon": 16,
                "forward_counts": {"vlm": 1, "dit": 4},
                "vlm_started_s": 100.0,
                "vlm_completed_s": 100.1,
                "action_started_s": 100.1,
                "action_completed_s": 100.2,
            },
        }
        if self.mutate:
            self.mutate(self.result)
        return self.result

    def close(self):
        assert threading.get_ident() == self.owner
        self.closed = True


def create_policy(**kwargs):
    from coexecution.libero_dynamic.policy import GrootChunkPolicy

    clients, prepared = [], []

    def factory(**options):
        client = Client(kwargs.pop("identity", None), kwargs.pop("mutate", None))
        clients.append(client)
        return client

    def preprocess(raw, language):
        prepared.append((raw, language))
        return {"sentinel": raw["frame"], "language": language}

    policy = GrootChunkPolicy(
        client_factory=factory,
        preprocessor=preprocess,
        **{k: v for k, v in kwargs.items() if k not in {"identity", "mutate"}},
    )
    return policy, clients, prepared


def test_default_fresh_flow4_prefix_and_exact_gripper_conversion():
    policy, clients, prepared = create_policy()
    assert clients == []
    raw = {"frame": 11}
    first = policy.get_action_chunk(raw, "pick the cup")
    second = policy.get_action_chunk({"frame": 22}, "place the cup")
    client = clients[0]
    assert prepared == [(raw, "pick the cup"), ({"frame": 22}, "place the cup")]
    assert first.shape == second.shape == (8, 7)
    np.testing.assert_array_equal(first[:, 6], [1, -1] * 4)
    np.testing.assert_array_equal(first[:, 0], np.arange(8, dtype=np.float32) / 20)
    assert [c[0] for c in client.calls] == [
        "identity",
        "reset",
        "synchronous_plan",
        "synchronous_plan",
    ]
    assert client.calls[-1][1]["observation"]["sentinel"] == 22
    record = policy.records[-1]
    assert record["decoded_horizon"] == 16
    assert record["execution_prefix_steps"] == 8
    assert record["source_tick_kind"] == "request_sequence"
    assert record["server_audit"]["forward_counts"] == {"vlm": 1, "dit": 4}
    assert record["model_actions"].shape == (16, 7)
    np.testing.assert_array_equal(record["model_actions"][:, 6], np.arange(16) % 2)
    client.result["actions"][:] = 99
    assert policy.records[-1]["model_actions"][0, 6] == 0
    first[:] = 99
    assert policy.records[0]["actions"][0, 6] == 1
    policy.close()
    assert client.closed


def test_reset_queues_seed_once_per_generation_and_close_owns_thread():
    policy, clients, _ = create_policy(seed=17)
    policy.reset(seed=23)
    assert clients == []
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(policy.get_action_chunk, {"frame": 1}, "one").result()
        pool.submit(policy.get_action_chunk, {"frame": 2}, "two").result()
        policy.reset(seed=29)
        pool.submit(policy.get_action_chunk, {"frame": 3}, "three").result()
        resets = [data for endpoint, data in clients[0].calls if endpoint == "reset"]
        assert resets == [{"options": {"seed": 23}}, {"options": {"seed": 29}}]
        with pytest.raises(RuntimeError, match="owning thread"):
            policy.close()
        assert not clients[0].closed
        pool.submit(policy.close).result()
    assert clients[0].closed
    with pytest.raises(RuntimeError, match="closed"):
        policy.get_action_chunk({"frame": 4}, "four")


def test_prepare_applies_pending_reset_on_owner_before_timed_inference():
    policy, clients, preprocessed = create_policy(seed=17)
    with ThreadPoolExecutor(max_workers=1) as pool:
        identity = pool.submit(policy.prepare).result()
        assert identity["predicted_horizon"] == 40
        identity["decoded_horizon"] = 99
        assert policy.identity["decoded_horizon"] == 16
        pool.submit(policy.prepare).result()
        assert [name for name, _ in clients[0].calls] == ["identity", "reset"]
        assert preprocessed == [] and policy.records == []
        with pytest.raises(RuntimeError, match="owning thread"):
            policy.prepare()
        policy.reset(seed=29)
        pool.submit(policy.prepare).result()
        pool.submit(policy.get_action_chunk, {"frame": 1}, "cup").result()
        assert [data for name, data in clients[0].calls if name == "reset"] == [
            {"options": {"seed": 17}},
            {"options": {"seed": 29}},
        ]
        assert "reset_started_s" not in policy.records[-1]
        pool.submit(policy.close).result()
    with pytest.raises(RuntimeError, match="closed"):
        policy.prepare()


def test_prepare_detects_reset_racing_with_barrier():
    policy, clients, _ = create_policy()
    started, finish = threading.Event(), threading.Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(policy.prepare).result()
        original_call = clients[0].call_endpoint

        def block_reset(endpoint, data=None, requires_input=True):
            if endpoint == "reset":
                started.set()
                assert finish.wait(5)
            return original_call(endpoint, data, requires_input)

        clients[0].call_endpoint = block_reset
        policy.reset(seed=29)
        pending = pool.submit(policy.prepare)
        assert started.wait(5)
        policy.reset(seed=31)
        finish.set()
        with pytest.raises(RuntimeError, match="reset during preparation"):
            pending.result()
        pool.submit(policy.prepare).result()
        assert clients[0].calls[-1] == ("reset", {"options": {"seed": 31}})
        assert policy.records == []
        pool.submit(policy.close).result()


def test_observation_metadata_preserves_physical_tick_and_simulation_clock():
    policy, clients, _ = create_policy()
    policy.get_action_chunk(
        {
            "frame": 1,
            "__pir2_capture": {"source_tick": 37, "capture_s": 18.0, "sim_time_s": 2.6},
        },
        "cup",
    )
    call = clients[0].calls[-1][1]
    assert call["source_tick"] == 37 and call["capture_s"] == 18.0
    record = policy.records[-1]
    assert record["source_tick_kind"] == "physical_tick"
    assert record["sim_time_s"] == 2.6
    policy.close()


def test_action_step_capture_is_not_mislabeled_as_physical_tick():
    policy, _clients, _ = create_policy()
    policy.get_action_chunk(
        {
            "frame": 1,
            "__pir2_capture": {
                "source_tick": 8,
                "source_tick_kind": "action_step",
                "capture_s": 18.0,
                "sim_time_s": 3.01,
            },
        },
        "cup",
    )
    assert policy.records[-1]["source_tick_kind"] == "action_step"
    assert policy.records[-1]["source_tick"] == 8
    assert policy.records[-1]["sim_time_s"] == 3.01
    policy.close()


@pytest.mark.parametrize(
    "metadata",
    [
        {"source_tick": 8, "source_tick_kind": "unknown_clock"},
        {"source_tick_kind": "action_step"},
    ],
)
def test_capture_rejects_ambiguous_tick_domain(metadata):
    policy, _clients, _ = create_policy()
    with pytest.raises(ValueError, match="source tick kind"):
        policy.get_action_chunk({"frame": 1, "__pir2_capture": metadata}, "cup")
    policy.close()


@pytest.mark.parametrize(
    "identity",
    [
        {"variant": "pir2"},
        {"role": "vlm"},
        {"server_layout": "paired"},
        {"paired_same_process": True},
        {"decoded_horizon": 7},
        {"predicted_horizon": 16},
        {"execution_endpoints": []},
    ],
)
def test_non_default_protocol_identity_rejected_before_plan(identity):
    policy, clients, _ = create_policy(identity=identity)
    with pytest.raises(ValueError):
        policy.get_action_chunk({"frame": 1}, "cup")
    assert [name for name, _ in clients[0].calls] == ["identity"]
    policy.close()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda result: result.update(actions=np.zeros((16, 6))),
        lambda result: result.update(actions=np.zeros((7, 7))),
        lambda result: result["actions"].__setitem__((0, 1), np.nan),
        lambda result: result["actions"].__setitem__((0, 6), -1),
        lambda result: result["audit"].update(forward_counts={"vlm": 0, "dit": 4}),
        lambda result: result["audit"].update(forward_counts={"vlm": 1, "dit": 1}),
        lambda result: result["audit"].update(decoded_horizon=40),
        lambda result: result["audit"].update(source_tick=999),
        lambda result: result["audit"].update(action_completed_s=99),
    ],
)
def test_invalid_model_result_rejected(mutate):
    policy, _clients, _ = create_policy(mutate=mutate)
    with pytest.raises(ValueError):
        policy.get_action_chunk({"frame": 1}, "cup")
    assert policy.records[-1]["status"] == "error"
    assert "server_audit" in policy.records[-1]
    policy.close()


def test_native_preprocessor_uses_official_holder_and_nested_shapes(monkeypatch):
    import sys

    from coexecution.libero_dynamic.policy import prepare_groot_observation

    calls = []

    class Official:
        def _process_observation(self, raw):
            calls.append((self._task_description, raw))
            raw["robot0_eef_quat"][3] = 1
            return {
                "video.image": raw["agentview_image"][::-1, ::-1],
                "video.wrist_image": raw["robot0_eye_in_hand_image"][::-1, ::-1],
                "state.x": [0.25],
                "state.gripper": [0.1, 0.2],
                "annotation.human.action.task_description": self._task_description,
            }

    monkeypatch.setitem(
        sys.modules,
        "gr00t.eval.sim.LIBERO.libero_env",
        SimpleNamespace(LiberoEnv=Official),
    )
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    raw = {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image + 10,
        "robot0_eef_quat": np.array([0, 0, 0, 1.1]),
    }
    nested = prepare_groot_observation(raw, "move cup")
    assert calls[0][0] == "move cup"
    assert raw["robot0_eef_quat"][3] == 1.1
    np.testing.assert_array_equal(nested["video"]["image"][0, 0], image[::-1, ::-1])
    np.testing.assert_array_equal(
        nested["video"]["wrist_image"][0, 0], (image + 10)[::-1, ::-1]
    )
    assert nested["state"]["gripper"].shape == (1, 1, 2)
    assert nested["language"]["annotation.human.action.task_description"] == [
        ["move cup"]
    ]


def test_reset_during_request_discards_previous_episode_result():
    policy, clients, _ = create_policy()
    started, finish = threading.Event(), threading.Event()

    def block(result):
        started.set()
        assert finish.wait(5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        # Construct the connection on its eventual owner, then block the next RPC.
        pool.submit(policy.get_action_chunk, {"frame": 0}, "before").result()
        clients[0].mutate = block
        pending = pool.submit(policy.get_action_chunk, {"frame": 1}, "old episode")
        assert started.wait(5)
        policy.reset(seed=91)
        finish.set()
        with pytest.raises(RuntimeError, match="discard old episode"):
            pending.result()
        assert policy.records[-1]["status"] == "error"
        clients[0].mutate = None
        pool.submit(policy.get_action_chunk, {"frame": 2}, "new episode").result()
        assert policy.records[-1]["seed"] == 91
        resets = [data for endpoint, data in clients[0].calls if endpoint == "reset"]
        assert resets[-1] == {"options": {"seed": 91}}
        pool.submit(policy.close).result()


def test_request_cannot_move_to_different_live_thread():
    policy, _clients, _ = create_policy()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(policy.get_action_chunk, {"frame": 1}, "cup").result()
        with pytest.raises(RuntimeError, match="owning thread"):
            policy.get_action_chunk({"frame": 2}, "cup")
        assert len(policy.records) == 1
        pool.submit(policy.close).result()


def test_close_before_any_request_does_not_construct_client():
    policy, clients, _ = create_policy()
    policy.close()
    policy.close()
    assert clients == []
    with pytest.raises(RuntimeError, match="closed"):
        policy.reset()


def test_timeout_socket_replacement_closes_old_socket_before_losing_reference(
    monkeypatch,
):
    import sys

    from coexecution.libero_dynamic.policy import _client_factory

    class Socket:
        def __init__(self):
            self.closed = False

        def close(self, *, linger):
            assert linger == 0
            self.closed = True

    class TimeoutRecreatingClient:
        def __init__(self, **options):
            self.context = SimpleNamespace(destroy=lambda **options: None)
            self._init_socket()

        def _init_socket(self):
            self.socket = Socket()

    monkeypatch.setitem(
        sys.modules,
        "gr00t.policy.server_client",
        SimpleNamespace(PolicyClient=TimeoutRecreatingClient),
    )
    client = _client_factory()
    old_socket = client.socket
    client._init_socket()  # Upstream invokes this after a REQ timeout.
    assert old_socket.closed
    assert not client.socket.closed
    client.close()
    assert client.socket.closed


@pytest.mark.parametrize("prefix", [1, 2])
def test_short_prefix_preserves_native_h16_and_full_flow4(prefix):
    policy, clients, _ = create_policy(chunk_steps=prefix)
    try:
        result = policy.get_action_chunk({"frame": 11}, "pick cup")
        assert result.shape == (prefix, 7)
        record = policy.records[-1]
        assert record["predicted_horizon"] == 40
        assert record["decoded_horizon"] == 16
        assert record["model_actions"].shape == (16, 7)
        assert record["execution_prefix_steps"] == prefix
        assert record["server_audit"]["forward_counts"] == {"vlm": 1, "dit": 4}
        assert clients[0].calls[-1][0] == "synchronous_plan"
    finally:
        policy.close()
