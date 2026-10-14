"""Flow4 scheduling and evidence contracts; never load a neural framework."""

import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from coexecution.profiling import flow
from coexecution.profiling.flow_protocol import (
    FlowCase,
    FlowSchedule,
    difference,
    validate_flow_head,
)


def arguments(*extra):
    return flow.parser().parse_args(
        [
            "--checkpoint",
            "/tmp/model",
            "--dataset",
            "/tmp/data",
            "--output",
            "/tmp/output",
            *extra,
        ]
    )


def test_import_and_help_do_not_load_tensor_framework():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from coexecution.profiling import flow; import coexecution.profiling.flow_execution; assert not ({'torch', 'gr00t', 'triton'} & set(sys.modules))",
        ],
        check=True,
    )


@pytest.mark.parametrize("precision", ["bf16", "w8a8", "w4a4"])
def test_common_graph_flags_and_component_precision_match(precision):
    args = arguments("--precision", precision)
    vlm, dit = (flow.configuration(args, role) for role in ("vlm", "dit"))
    assert vlm.precision == dit.precision == precision
    assert vlm.vlm_graph and not vlm.action_graph
    assert dit.action_graph and dit.action_kv_reuse and not dit.vlm_graph
    assert all(c.compiled_host_guards and c.input_preparation for c in (vlm, dit))
    if precision == "w8a8":
        assert vlm.operator_profile.name == "w8-splitk-full.json"
        assert dit.tactics.name == "site-tactics.json"
        assert dit.group_tactics.name == "group-tactics.json"
        assert dit.shape_tactics.name == "w8-shape-tactics.json"
        assert not dit.norm_modulation_quant
    elif precision == "w4a4":
        assert dit.operator_profile is None
        assert (
            dit.norm_modulation_quant
            and dit.residual_norm_quant
            and dit.group_conditioning
        )
    else:
        assert dit.operator_profile is None and not dit.group_conditioning


def test_cooperative_controls_disable_whole_vlm_graph_and_isolate_fences():
    args = arguments("--group", "cooperative")
    assert not flow.configuration(args, "vlm").vlm_graph
    cases = flow.cases(args)
    names = {case.mode for case in cases}
    assert {
        "vlm_only",
        "vlm_only_fenced",
        "serial",
        "serial_fenced",
        "concurrent_equal_fenced",
        "concurrent_dit_priority_fenced",
        "cooperative_dit_priority",
    } <= names
    for mode in (
        "concurrent_equal",
        "concurrent_dit_priority",
        "cooperative_dit_priority",
    ):
        assert {case.offset_ms for case in cases if case.mode == mode} == {0.0, 5.0}
    assert FlowCase("vlm_only_fenced").fenced
    assert not FlowCase("vlm_only").fenced


def test_graph_yield_rejected_before_models_and_dual_device_explicit():
    with pytest.raises(ValueError, match="cooperative"):
        flow.cases(arguments("--modes", "cooperative_dit_priority"))
    args = arguments("--action-device", "cuda:0", "--vlm-device", "cuda:1")
    flow.validate_options(args)
    assert args.action_device != args.vlm_device
    for value in ("cpu", "cuda", "cuda:-1"):
        with pytest.raises(ValueError, match="device"):
            flow.validate_options(arguments("--action-device", value))
    for value in ("nan", "inf", "-1"):
        with pytest.raises(ValueError, match="offset"):
            flow.cases(arguments("--offsets-ms", value))


def test_official_flow_horizon_and_nfe_are_strict():
    from types import SimpleNamespace

    head = SimpleNamespace(
        config=SimpleNamespace(streaming=False, action_horizon=40),
        action_horizon=40,
        num_inference_timesteps=4,
    )
    validate_flow_head(head)
    head.num_inference_timesteps = 1
    with pytest.raises(ValueError, match="Flow4"):
        validate_flow_head(head)


def fake_result(value=1):
    start = time.perf_counter_ns()
    return {
        "output": value,
        "host_submit_start_ns": start,
        "host_submit_end_ns": start,
        "ready_ns": time.perf_counter_ns(),
        "gpu_span_ms": 1.0,
    }


def test_serial_waits_for_own_action_ready_and_keeps_old_generation():
    order = []

    def execute(role, case, index, gate, request_id):
        order.append(role)
        return fake_result()

    with FlowSchedule(execute) as schedule:
        row, outputs = schedule.pair(
            FlowCase("serial"), 1, 9, old_generation=20, next_generation=21
        )
    assert order == ["dit", "vlm"]
    assert row["s1_feature_generation"] == 20
    assert row["s2_feature_generation"] == 21
    assert row["s2_release_ns"] >= row["s1_ready_ns"]
    assert row["pair_makespan_ms"] >= row["s1_response_ms"]
    assert set(outputs) == {"dit", "vlm"}
    assert row["s1_request_id"] != row["s2_request_id"]
    assert "event_span_intersection_ms" not in row


def test_cooperative_response_includes_completed_boundary_wait():
    submitted = threading.Event()

    def execute(role, case, index, gate, request_id):
        if role == "vlm":
            assert gate.requested.wait(1)
            # This fake executor models CUDA completion before handing off.
            time.sleep(0.01)
            assert not submitted.is_set()
            gate.boundary("completed-layer")
            assert submitted.is_set()
        else:
            submitted.set()
        return fake_result()

    with FlowSchedule(execute, timeout_s=2) as schedule:
        row, _ = schedule.pair(
            FlowCase("cooperative_dit_priority"),
            0,
            1,
            old_generation=0,
            next_generation=1,
        )
    assert row["s1_pre_submit_wait_ms"] >= 8
    assert row["s1_response_ms"] >= row["s1_pre_submit_wait_ms"]
    events = [entry["event"] for entry in row["gate_events"]]
    assert (
        events.index("vlm_yield")
        < events.index("dit_complete")
        < events.index("vlm_resume")
    )


def test_failure_aborts_other_worker_before_pair_returns():
    idle = threading.Event()

    def execute(role, case, index, gate, request_id):
        if role == "vlm":
            try:
                gate.wait(gate.dit_done, "DiT completion")
            finally:
                idle.set()
        else:
            raise ValueError("diagnostic failure")
        return fake_result()

    with FlowSchedule(execute, timeout_s=1) as schedule:
        with pytest.raises((ValueError, RuntimeError)):
            schedule.pair(
                FlowCase("concurrent_equal"), 0, 0, old_generation=0, next_generation=1
            )
        assert idle.is_set()


def test_numerical_drift_is_reported_without_exact_equality_veto():
    left = {"action_pred": np.array([1.0, 2.0])}
    value = difference(left, {"action_pred": np.array([1.5, 1.5])})
    assert value["passes"] and not value["bitwise_equal"]
    assert value["mae"] == 0.5 and value["max_abs"] == 0.5
    assert value["relative_l2"] == pytest.approx((0.5 / 5) ** 0.5)
    for candidate in (
        {"bad": np.ones(2)},
        {"action_pred": np.ones(3)},
        {"action_pred": np.array([1, np.nan])},
    ):
        with pytest.raises(ValueError):
            difference(left, candidate)


def test_quantization_profiles_are_repository_relative():
    args = arguments("--precision", "w8a8")
    config = flow.configuration(args, "dit")
    assert config.operator_profile.is_relative_to(Path(flow.ROOT))


def test_close_offsets_keep_distinct_case_and_request_cohorts():
    plan = flow.cases(
        arguments(
            "--modes", "concurrent_equal", "--offsets-ms", "1.0000001", "1.0000002"
        )
    )
    assert len({case.identity for case in plan}) == 2
    for case in plan:
        assert float(case.identity.split("@")[1][:-2]) == case.offset_ms

    def execute(*_args):
        return fake_result()

    with FlowSchedule(execute) as schedule:
        rows = [
            schedule.pair(case, 0, 0, old_generation=0, next_generation=1)[0]
            for case in plan
        ]
    assert len({row["case"] for row in rows}) == 2
    assert len({row["s1_request_id"] for row in rows}) == 2
    assert len({row["s2_request_id"] for row in rows}) == 2
    assert [row["offset_ms"] for row in rows] == [1.0000001, 1.0000002]
    assert FlowCase("concurrent_equal", 0).identity == "concurrent_equal@0ms"
    assert FlowCase("concurrent_equal", 5).identity == "concurrent_equal@5ms"


def test_warmup_uses_the_same_persistent_worker_as_measurement():
    owners = []

    def execute(role, case, index, gate, request_id):
        owners.append(threading.get_ident())
        return fake_result()

    with FlowSchedule(execute) as schedule:
        schedule.warm("dit", FlowCase("dit_only"), 0)
        schedule.pair(FlowCase("dit_only"), 0, 0, old_generation=0, next_generation=1)
    assert owners[0] == owners[1] != threading.get_ident()


def test_cuda_profile_range_is_explicit_and_only_for_separate_probe():
    args = arguments("--cuda-profiler-range", "--profile-iterations", "1")
    flow.validate_options(args)
    assert args.cuda_profiler_range
    with pytest.raises(ValueError, match="profile"):
        flow.validate_options(arguments("--cuda-profiler-range"))


def test_graph_audit_requires_every_replay_and_rejects_recapture():
    from copy import deepcopy

    from coexecution.profiling.flow_execution import graph_audit

    counts = {"capture_count": 1, "replay_count": 4, "failure_count": 0}
    before = {
        "vlm": {"vlm": counts, "dit": None, "action": None},
        "dit": {"vlm": None, "dit": None, "action": dict(counts)},
    }
    after = deepcopy(before)
    after["vlm"]["vlm"]["replay_count"] += 3
    after["dit"]["action"]["replay_count"] += 2
    graph_audit(before, after, vlm_calls=3, dit_calls=2)
    after["dit"]["action"]["capture_count"] += 1
    with pytest.raises(RuntimeError, match="capture"):
        graph_audit(before, after, vlm_calls=3, dit_calls=2)


def test_completed_boundary_fences_cuda_before_permitting_dit():
    from coexecution.profiling.flow_workload import FlowWorkload

    order = []
    gate = SimpleNamespace(boundary=lambda name: order.append(("permit", name)))
    fake = SimpleNamespace(
        context=SimpleNamespace(
            current=(FlowCase("cooperative_dit_priority"), gate, [0])
        ),
        args=SimpleNamespace(vlm_chunk_blocks=1),
        layer_modules=[("layer", None)],
        streams={
            "vlm": SimpleNamespace(synchronize=lambda: order.append("cuda-complete"))
        },
        torch=SimpleNamespace(
            cuda=SimpleNamespace(nvtx=SimpleNamespace(range=lambda _: nullcontext()))
        ),
    )
    FlowWorkload._boundary(fake, "layer")
    assert order == ["cuda-complete", ("permit", "layer")]


def test_integer_coverage_observes_group_pack_and_category_dispatch():
    from coexecution.profiling.flow_workload import ProjectionEvidence

    linear = SimpleNamespace(bits=8)
    group = SimpleNamespace(linear=linear)
    category = SimpleNamespace(bits=8, packed_weight=object())

    class Controller:
        def __init__(self):
            self.quant = {8: {"action_head.linear": linear}}
            self.grouped = {8: [(["action_head.linear"], group)]}

        def clear(self):
            pass

        def dispatch(self, q, value):
            return value

        def _pack_dispatch_input(self, q, value):
            return value

    controller = Controller()
    module = SimpleNamespace(forward=lambda x, q=category: x)
    session = SimpleNamespace(
        _clear_cache=controller.clear,
        coverage={
            "linears": ["action_head.linear"],
            "category_linears": ["action_head.category"],
        },
        policy=SimpleNamespace(
            model=SimpleNamespace(
                named_modules=lambda: [("action_head.category", module)]
            )
        ),
    )
    observer = ProjectionEvidence(session, "dit")
    controller._pack_dispatch_input(linear, 3)
    controller.dispatch(category, 4)
    assert observer.evidence()["passes"]
    before = dict(observer.counts)
    observer.active = False
    controller.dispatch(category, 4)
    assert observer.counts == before
    observer.close()


def test_integer_coverage_observes_real_candidate_residual_and_shape_tactic(
    monkeypatch,
):
    from coexecution.kernels import ampere_ada
    from coexecution.kernels.ampere_ada._candidate_support import _ProjectionTactic
    from coexecution.optimization.groot import integer_candidates
    from coexecution.profiling.flow_workload import ProjectionEvidence

    dtype = object()

    class TensorMetadata:
        device = "cuda:0"
        shape = (1, 512)

        def is_contiguous(self):
            return True

        def data_ptr(self):
            return 16

        def view(self, *_shape):
            return self

    residual = TensorMetadata()
    residual.dtype = dtype
    split_k = SimpleNamespace(split_k_gemm=lambda *_args, **_kwargs: residual)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(Tensor=TensorMetadata, bfloat16=dtype)
    )
    for name, module in (("epilogue", SimpleNamespace()), ("split_k", split_k)):
        monkeypatch.setattr(ampere_ada, name, module, raising=False)
        monkeypatch.setitem(
            sys.modules, f"coexecution.kernels.ampere_ada.{name}", module
        )
    monkeypatch.setattr(integer_candidates, "is_ada_device", lambda _device: True)
    projection = SimpleNamespace(
        bits=8,
        tactic=0,
        shape_tactics={1: 2},
        out_features=512,
        packed_weight=SimpleNamespace(device="cuda:0", shape=(512, 512)),
    )

    class Controller:
        def __init__(self):
            self.quant = {8: {"action_head.attn.to_out": projection}}
            self.grouped = {8: []}
            self.candidates = integer_candidates.IntegerCandidateDispatch(
                self, implementation="split_k"
            )

        def clear(self):
            pass

        def dispatch(self, q, value):
            return value

        def _pack_dispatch_input(self, q, value):
            return value

    controller = Controller()
    monkeypatch.setattr(controller.candidates, "_kernel", lambda _q, _rows: split_k)
    packed = SimpleNamespace(shape=(1,))
    activations = SimpleNamespace(
        _robotics_packed_input=True, for_bits=lambda _bits: packed
    )
    session = SimpleNamespace(
        _clear_cache=controller.clear,
        coverage={"linears": ["action_head.attn.to_out"]},
        policy=SimpleNamespace(model=SimpleNamespace(named_modules=list)),
    )
    original_pack = controller.candidates.pack_input
    observer = ProjectionEvidence(session, "dit")
    assert controller.candidates.residual(projection, activations, residual) is residual
    evidence = observer.evidence()
    assert evidence["passes"], evidence
    assert any(row["adapter"] == "candidate.pack_input" for row in evidence["observed"])
    controller.dispatch(_ProjectionTactic(projection, 2), 3)
    assert observer.evidence()["passes"]
    observer.close()
    assert controller.candidates.pack_input == original_pack


@pytest.mark.parametrize("drain_fails", [False, True])
def test_cuda_failure_aborts_peer_and_drains_owned_stream_before_teardown(drain_fails):
    from coexecution.profiling.flow_execution import FlowExecutor
    from coexecution.profiling.scheduling import CooperativeGate

    order = []
    gate = CooperativeGate()
    original_error = ValueError("forward failed after CUDA enqueue")

    def fail(*_args):
        order.append("enqueued")
        raise original_error

    def drain():
        assert gate.aborted.is_set()
        order.append("drain")
        if drain_fails:
            raise RuntimeError("drain also failed")

    stream = SimpleNamespace(synchronize=drain)
    fake_torch = SimpleNamespace(
        inference_mode=nullcontext,
        cuda=SimpleNamespace(
            device=lambda _: nullcontext(),
            stream=lambda _: nullcontext(),
            Event=lambda **_: SimpleNamespace(record=lambda _: None),
            nvtx=SimpleNamespace(range=lambda _: nullcontext()),
        ),
    )
    workload = SimpleNamespace(
        torch=fake_torch,
        streams={"dit_equal": stream},
        devices={"dit": "cuda:1"},
        fixtures=[None],
        context=threading.local(),
        reference_features=[{}],
        prepared={"dit": [SimpleNamespace(action={})]},
        engines={"dit": SimpleNamespace(action=fail)},
    )
    with pytest.raises(ValueError) as caught:
        FlowExecutor(workload)("dit", FlowCase("dit_only"), 0, gate, "failed-request")
    order.append("teardown")
    assert caught.value is original_error
    assert order == ["enqueued", "drain", "teardown"]
    assert workload.context.current is None


def test_pair_failure_drains_both_cuda_workers_before_model_teardown():
    from coexecution.profiling.flow_execution import FlowExecutor

    context = threading.local()
    vlm_enqueued = threading.Event()
    drained = []

    def action(*_args):
        assert vlm_enqueued.wait(1)
        raise ValueError("action enqueue failure")

    def vlm(*_args):
        vlm_enqueued.set()
        context.current[1].wait(threading.Event(), "peer abort")

    def drain(role):
        assert context.current[1].aborted.is_set()
        drained.append(role)

    fake_torch = SimpleNamespace(
        inference_mode=nullcontext,
        cuda=SimpleNamespace(
            device=lambda _: nullcontext(),
            stream=lambda _: nullcontext(),
            Event=lambda **_: SimpleNamespace(record=lambda _: None),
            nvtx=SimpleNamespace(range=lambda _: nullcontext()),
        ),
    )
    workload = SimpleNamespace(
        torch=fake_torch,
        streams={
            "dit_equal": SimpleNamespace(synchronize=lambda: drain("dit")),
            "vlm": SimpleNamespace(synchronize=lambda: drain("vlm")),
        },
        devices={"dit": "cuda:0", "vlm": "cuda:1"},
        fixtures=[None],
        context=context,
        reference_features=[{}],
        prepared={
            "dit": [SimpleNamespace(action={})],
            "vlm": [SimpleNamespace(backbone={})],
        },
        engines={
            "dit": SimpleNamespace(action=action),
            "vlm": SimpleNamespace(vlm=vlm),
        },
    )
    with FlowSchedule(FlowExecutor(workload), timeout_s=1) as schedule:
        with pytest.raises(ValueError, match="action enqueue"):
            schedule.pair(
                FlowCase("concurrent_equal"), 0, 0, old_generation=0, next_generation=1
            )
        assert sorted(drained) == ["dit", "vlm"]
    drained.append("model-teardown")
    assert drained[-1] == "model-teardown"


@pytest.mark.parametrize("static_adapter_active", [False, True])
def test_coverage_declares_only_installed_static_vlm_lm_head_inactive(
    static_adapter_active,
):
    from coexecution.optimization.groot.vlm import PolicyStaticS2
    from coexecution.profiling.flow_workload import ProjectionEvidence

    active = SimpleNamespace(bits=8)
    inactive = SimpleNamespace(bits=8)
    names = {"backbone.active_projection": active, "backbone.model.lm_head": inactive}

    class Controller:
        def __init__(self):
            self.quant = {8: names}
            self.grouped = {8: []}

        def clear(self):
            pass

        def dispatch(self, q, value):
            return value

        def _pack_dispatch_input(self, q, value):
            return value

    controller = Controller()
    adapter = object.__new__(PolicyStaticS2)
    adapter._entered = static_adapter_active
    session = SimpleNamespace(
        _clear_cache=controller.clear,
        config=SimpleNamespace(static_vlm=True),
        static_vlm=adapter,
        coverage={"linears": list(names)},
        policy=SimpleNamespace(model=SimpleNamespace(named_modules=list)),
    )
    observer = ProjectionEvidence(session, "vlm")
    # The active projection remains mandatory even when the LM head is excluded.
    assert not observer.evidence()["passes"]
    controller.dispatch(active, 1)
    result = observer.evidence()
    assert result["selected_projections"] == sorted(names)
    assert result["passes"] == static_adapter_active
    if static_adapter_active:
        assert result["expected_active_projections"] == ["backbone.active_projection"]
        assert (
            result["inactive_selected_projections"][0]["name"]
            == "backbone.model.lm_head"
        )
        assert (
            "StaticS2.forward_inputs"
            in result["inactive_selected_projections"][0]["evidence"]
        )
        controller.dispatch(inactive, 1)
        assert not observer.evidence()["passes"]
        assert observer.evidence()["unexpected_inactive_execution"] == [
            "backbone.model.lm_head"
        ]
    else:
        assert result["missing"] == ["backbone.model.lm_head"]
        controller.dispatch(inactive, 1)
        assert observer.evidence()["passes"]
    observer.close()
