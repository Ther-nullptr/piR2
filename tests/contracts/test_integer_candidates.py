"""Candidate selection and patch lifecycle without importing tensor libraries."""

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from coexecution.kernels.ampere_ada._candidate_support import projection_for_rows
from coexecution.optimization.groot.group_outputs import GroupOutputAdapters
from coexecution.optimization.groot.integer_candidates import IntegerCandidateDispatch


def test_default_candidate_does_not_inspect_or_prepare_inputs():
    candidate = IntegerCandidateDispatch(object())
    assert candidate.dispatch(object(), object()) is None


@pytest.mark.parametrize(
    "options",
    [
        {"implementation": "fastest"},
        {"small_m_tactic": 6},
        {"small_m_tactic": True},
        {"split_k_slices": 16},
        {"split_k_slices": True},
        {"split_k_tile": 3},
    ],
)
def test_invalid_candidate_options_fail_before_inspecting_controller(options):
    with pytest.raises(ValueError):
        IntegerCandidateDispatch(object(), **options)


def test_shape_specific_tactic_does_not_mutate_shared_projection():
    original = SimpleNamespace(tactic=0, shape_tactics={3: 7}, packed_weight=object())
    selected = projection_for_rows(original, 3)
    assert selected.tactic == 7
    assert selected.packed_weight is original.packed_weight
    assert original.tactic == 0
    assert projection_for_rows(original, 1) is original


def test_ineligible_dispatch_returns_before_packing(monkeypatch):
    candidate = IntegerCandidateDispatch(object(), implementation="small_m")
    projection = SimpleNamespace(tactic=0)
    observed = []
    monkeypatch.setattr(
        candidate, "_kernel", lambda q, rows: observed.append(rows) or None
    )
    monkeypatch.setattr(
        candidate, "pack_input", lambda *args: pytest.fail("ineligible input packed")
    )
    assert candidate.dispatch(projection, SimpleNamespace(shape=(2, 3, 128))) is None
    assert observed == [6]


def test_candidate_crops_physical_padding_and_restores_leading_shape(monkeypatch):
    candidate = IntegerCandidateDispatch(
        object(), implementation="small_m", small_m_tactic=3
    )
    projection = SimpleNamespace(tactic=0, shape_tactics={6: 7}, out_features=15)
    packed = SimpleNamespace(shape=(2, 3))
    calls = []

    class Matrix:
        def __getitem__(self, index):
            assert index == (slice(None), slice(None, 15))
            return self

        def reshape(self, *shape):
            return shape

    def small_m(p, q, tactic):
        calls.append((p, q.tactic, tactic))
        return Matrix()

    monkeypatch.setattr(
        candidate, "_kernel", lambda q, rows: SimpleNamespace(small_m_gemm=small_m)
    )
    monkeypatch.setattr(candidate, "pack_input", lambda *args: packed)
    assert candidate.dispatch(projection, SimpleNamespace(shape=(2, 3, 128))) == (
        2,
        3,
        15,
    )
    assert calls == [(packed, 7, 3)]
    assert projection.tactic == 0


class _Group:
    contiguous_outputs = True
    sizes = (8, 8)
    linear = object()

    def __init__(self):
        self._input = self._output = None
        self._remaining = set()

    def project(self, x, index):
        return "reference"


def _group_adapter(monkeypatch):
    native = ModuleType("coexecution.kernels.ampere_ada.epilogue")
    calls = []

    def outputs(packed, group):
        calls.append(packed)
        return object(), object()

    native.group_outputs = outputs
    monkeypatch.setitem(sys.modules, native.__name__, native)
    group = _Group()
    controller = SimpleNamespace(
        grouped={4: [(("q", "k"), group)]},
        _pack_dispatch_input=lambda q, x, lut, up: x,
    )
    return GroupOutputAdapters(controller), group, calls, native


def test_group_candidates_pack_once_per_sibling_cycle_and_restore(monkeypatch):
    adapter, group, calls, _ = _group_adapter(monkeypatch)
    adapter.set(4)
    first = object()
    output = group.project(first, 1)
    assert group.project(first, 0) is not output
    assert calls == [first]
    assert group._input is group._output is None
    assert not group._remaining
    second = object()
    group.project(second, 0)
    assert calls == [first, second]
    adapter.close()
    assert "project" not in group.__dict__
    assert group.project(first, 0) == "reference"
    assert group._input is group._output is None
    assert not group._remaining


def test_group_failure_clears_partial_cycle_and_close_is_idempotent(monkeypatch):
    adapter, group, _, native = _group_adapter(monkeypatch)
    adapter.set(4)
    group.project(object(), 0)

    def fail(*args):
        raise RuntimeError("native writer failed")

    native.group_outputs = fail
    with pytest.raises(RuntimeError, match="native writer failed"):
        group.project(object(), 1)
    assert group._input is group._output is None
    assert not group._remaining
    adapter.close()
    adapter.close()
    assert "project" not in group.__dict__


def test_group_writer_preserves_an_explicit_gemm_candidate(monkeypatch):
    adapter, group, calls, _ = _group_adapter(monkeypatch)
    group.offsets = (0, 8)
    dispatched = []

    class Output:
        def narrow(self, dimension, offset, size):
            return SimpleNamespace(contiguous=lambda: (dimension, offset, size))

    adapter.controller.candidates = SimpleNamespace(supports=lambda q, x: True)
    adapter.controller.dispatch = lambda q, x: dispatched.append(x) or Output()
    adapter.set(4)
    x = object()
    assert group.project(x, 0) == (-1, 0, 8)
    assert group.project(x, 1) == (-1, 8, 8)
    assert dispatched == [x]
    assert calls == []
    assert not group._remaining


def test_partial_group_installation_failure_restores_preceding_group(monkeypatch):
    adapter, group, _, _ = _group_adapter(monkeypatch)

    class InvalidGroup(_Group):
        @property
        def project(self):
            raise RuntimeError("invalid group")

    adapter.controller.grouped[4].append((("x", "y"), InvalidGroup()))
    with pytest.raises(RuntimeError, match="invalid group"):
        adapter.set(4)
    assert "project" not in group.__dict__
    assert not adapter.saved


def test_source_templates_reject_mutated_owned_source_without_tensor_imports(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

torch = ModuleType("torch")
torch.cuda = SimpleNamespace(
    is_initialized=lambda: True,
    is_current_stream_capturing=lambda: True,
)
sys.modules["torch"] = torch
from coexecution.kernels.ampere_ada import epilogue, int4_small_m
from coexecution.kernels.ampere_ada._candidate_support import require_pre_capture_build

source = Path("coexecution/kernels/ampere_ada/csrc/integer_gemm.cu")
original = source.read_bytes()
inputs = {"sources": [source], "cutlass_revision": epilogue._CUTLASS_REVISION}
for module in (epilogue, int4_small_m):
    header = module._pinned_run_header(inputs)
    assert header.startswith(b"#pragma once")
    assert b"TORCH_LIBRARY" not in header
    module._cache_run_header(Path(sys.argv[1]), header)
    assert (Path(sys.argv[1]) / module._CACHED_HEADER).read_bytes() == header
mutated = Path(sys.argv[1]) / "mutated.cu"
mutated.write_bytes(original + b"// changed")
inputs["sources"] = [mutated]
for module in (epilogue, int4_small_m):
    try:
        module._pinned_run_header(inputs)
    except RuntimeError as error:
        assert "SHA256" in str(error)
    else:
        raise AssertionError("mutated source accepted")
assert source.read_bytes() == original
try:
    require_pre_capture_build()
except RuntimeError as error:
    assert "before CUDA Graph capture" in str(error)
else:
    raise AssertionError("graph capture did not reject JIT work")
assert "torch.utils.cpp_extension" not in sys.modules
assert "triton" not in sys.modules
""",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_text_rms_producer_without_pointwise_restores_original_methods():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from types import ModuleType, SimpleNamespace

# Contract-only objects: no tensors, model library or CUDA runtime is loaded.
sys.modules["torch"] = ModuleType("torch")
modulation = ModuleType("coexecution.kernels.ampere_ada.modulation")
class Packs:
    def __init__(self, values):
        self.values = values
modulation.PackedActivations = Packs
sys.modules[modulation.__name__] = modulation
native = ModuleType("coexecution.kernels.ampere_ada.producers")
native.prepare_rmsnorm = lambda x, weight, eps, bits: (x, weight, eps, bits)
native.prepare_final_modulation = lambda *args: None
sys.modules[native.__name__] = native
linear = ModuleType("coexecution.optimization.backends.integer.linear")
linear.prepare = lambda *args: None
sys.modules[linear.__name__] = linear
from coexecution.optimization.groot.producers import QuantizationProducers

class Norm:
    weight = object()
    variance_epsilon = 1e-6
    def forward(self, x):
        return x
layer = type("Qwen3VLTextDecoderLayer", (), {})()
layer.input_layernorm = Norm()
layer.post_attention_layernorm = Norm()
layer.self_attn = SimpleNamespace(q_proj=object(), k_proj=object(), v_proj=object())
layer.mlp = SimpleNamespace(gate_proj=object(), up_proj=object())
modules = [*vars(layer.self_attn).values(), *vars(layer.mlp).values()]
controller = SimpleNamespace(sites={str(i): (module,) for i, module in enumerate(modules)})
policy = SimpleNamespace(model=SimpleNamespace(named_modules=lambda: [("decoder", layer)]))
adapter = QuantizationProducers(policy, controller, None)
assert not adapter.final_selected
adapter.set(4, final_modulation=False)
value = SimpleNamespace(shape=(1, 2, 8))
for norm in (layer.input_layernorm, layer.post_attention_layernorm):
    result = norm.forward(value)
    assert result.values == {4: (value, norm.weight, norm.variance_epsilon, 4)}
    assert result.shape == value.shape
assert len(adapter.evidence()["rmsnorm_sites"]) == 2
assert not adapter.evidence()["final_modulation"]
adapter.close()
adapter.close()
for norm in (layer.input_layernorm, layer.post_attention_layernorm):
    assert "forward" not in norm.__dict__
    assert norm.forward(value) is value
assert not adapter.evidence()["rmsnorm_sites"]
projection = object()
controller.sites["final"] = (projection,)
original_final = object()
pointwise = SimpleNamespace(
    model=SimpleNamespace(proj_out_2=projection),
    final_projection_prepare=original_final,
)
adapter = QuantizationProducers(policy, controller, pointwise)
adapter.set(4, rmsnorm=False, final_modulation=True)
assert adapter.evidence()["rmsnorm_sites"] == []
assert adapter.evidence()["final_modulation"]
assert pointwise.final_projection_prepare is not original_final
for norm in (layer.input_layernorm, layer.post_attention_layernorm):
    assert "forward" not in norm.__dict__
adapter.close()
assert pointwise.final_projection_prepare is original_final
assert "triton" not in sys.modules
""",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_gemv_and_norm_reject_stale_registrations_and_failed_build_retries():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

torch = ModuleType("torch")
torch.ops = SimpleNamespace()
torch.cuda = SimpleNamespace(is_initialized=lambda: False)
torch.nn = ModuleType("torch.nn")
torch.nn.functional = ModuleType("torch.nn.functional")
sys.modules.update({"torch": torch, "torch.nn": torch.nn, "torch.nn.functional": torch.nn.functional})
integer = ModuleType("coexecution.kernels.ampere_ada.integer")
integer.PackedActivation = object
sys.modules[integer.__name__] = integer
linear = ModuleType("coexecution.optimization.backends.integer.linear")
linear.prepare = lambda *args: None
sys.modules[linear.__name__] = linear
from coexecution.kernels.ampere_ada import int4_gemv, norm

for module, source, namespace, operation, loader_name in (
    (int4_gemv, "int4_gemv", "pir2_int4_gemv", "gemv", "load_gemv_extension"),
    (norm, "layernorm_pack", "pir2_layernorm_pack", "prepare", "load_norm_extension"),
):
    module.owned_build_identity = lambda: b"owned"
    loader = getattr(module, loader_name)
    extension = SimpleNamespace(**{operation: lambda *args: None})
    setattr(torch.ops, namespace, extension)
    for missing_or_stale in (None, "old"):
        if missing_or_stale is not None:
            extension.source_identity = lambda: "old"
        try:
            loader()
        except RuntimeError as error:
            assert "identity differs" in str(error)
        else:
            raise AssertionError("stale registration accepted")
    expected = hashlib.sha256((Path(module.__file__).parent / "csrc" / (source + ".cu")).read_bytes() + b"\\0owned").hexdigest()
    extension.source_identity = lambda: expected
    assert loader() is extension
    loader.cache_clear()
    setattr(torch.ops, namespace, SimpleNamespace())
    jit = ModuleType("torch.utils.cpp_extension")
    builds = []
    def fail(**kwargs):
        builds.append(kwargs["name"])
        raise RuntimeError("compiler failed")
    jit.load = fail
    sys.modules[jit.__name__] = jit
    try:
        loader()
    except RuntimeError as error:
        assert "compiler failed" in str(error)
    else:
        raise AssertionError("compilation unexpectedly succeeded")
    try:
        loader()
    except RuntimeError as error:
        assert "previously failed" in str(error)
    else:
        raise AssertionError("failed identity retried")
    assert len(builds) == 1
""",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
