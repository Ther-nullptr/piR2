"""Host-side guard and signature contracts; no numerical model execution."""

import importlib
import importlib.util
import inspect
from collections import Counter, OrderedDict
from types import SimpleNamespace

import pytest

MODULE = "coexecution.optimization.groot.vlm"


def apply_rotary_pos_emb(value):
    return value


def implementation():
    assert importlib.util.find_spec(MODULE) is not None, (
        "Static policy VLM module is missing"
    )
    return importlib.import_module(MODULE)


class TensorMetadata:
    def __init__(self, shape=(8, 8), *, dtype="bf16", device="cuda:0"):
        self.shape, self.dtype, self.device = shape, dtype, device
        self._version = 0
        self.requires_grad = True
        self.layout = "strided"
        self._stride = (shape[-1], 1)

    def stride(self):
        return self._stride


class ModelMetadata:
    def __init__(self):
        self.training = False
        self.config = SimpleNamespace(_attn_implementation="sdpa", use_cache=False)
        self.weight = TensorMetadata()
        self.buffer = TensorMetadata((1, 8))
        self._forward_hooks = {}
        self._forward_pre_hooks = {}
        self.forward = lambda values: values

    def named_modules(self):
        return iter([("", self)])

    def named_parameters(self):
        return iter([("weight", self.weight)])

    def named_buffers(self):
        return iter([("buffer", self.buffer)])


def test_policy_static_api_is_modular_and_graph_is_explicitly_opt_in():
    cls = implementation().PolicyStaticS2
    parameters = inspect.signature(cls).parameters
    assert parameters["max_entries"].default == 32
    assert parameters["cuda_graph"].default is False
    assert parameters["vision_rope"].default is None
    for method in ("__enter__", "close", "close_guard", "evidence"):
        assert callable(getattr(cls, method, None))


@pytest.mark.parametrize("changed", ["weight", "buffer"])
def test_execution_stamp_tracks_tensor_versions_and_replacement(changed):
    stamp = implementation().execution_stamp
    model = ModelMetadata()
    before = stamp(model)
    getattr(model, changed)._version += 1
    assert stamp(model) != before
    before = stamp(model)
    setattr(model, changed, TensorMetadata())
    assert stamp(model) != before


@pytest.mark.parametrize("changed", ["shape", "dtype", "device", "_stride"])
def test_execution_stamp_tracks_tensor_layout_without_device_reads(changed):
    stamp = implementation().execution_stamp
    model = ModelMetadata()
    before = stamp(model)
    setattr(
        model.weight,
        changed,
        {"shape": (4, 16), "dtype": "fp32", "device": "cuda:1", "_stride": (1, 8)}[
            changed
        ],
    )
    assert stamp(model) != before


@pytest.mark.parametrize(
    "changed", ["training", "backend", "forward", "hook", "pre_hook"]
)
def test_execution_stamp_tracks_captured_callables_backends_and_hooks(changed):
    stamp = implementation().execution_stamp
    model = ModelMetadata()
    before = stamp(model)
    if changed == "training":
        model.training = True
    elif changed == "backend":
        model.config._attn_implementation = "eager"
    elif changed == "forward":
        model.forward = lambda values: (values, "new")
    elif changed == "hook":
        model._forward_hooks[10] = lambda *args: None
    else:
        model._forward_pre_hooks[10] = lambda *args: None
    assert stamp(model) != before


def test_execution_stamp_accepts_frozen_eval_requires_grad_flags():
    stamp = implementation().execution_stamp
    model = ModelMetadata()
    assert model.weight.requires_grad
    assert stamp(model) == stamp(model)


def test_execution_stamp_tracks_qwen_global_bindings_without_editing_them(monkeypatch):
    stamp = implementation().execution_stamp
    model = ModelMetadata()

    def forward(values):
        return apply_rotary_pos_emb(values)

    model.forward = forward
    before = stamp(model)
    monkeypatch.setitem(
        forward.__globals__, "apply_rotary_pos_emb", lambda values: (values, "new")
    )
    assert stamp(model) != before


@pytest.mark.parametrize("changed", ["packed_weight", "weight_scale", "tactic"])
def test_execution_stamp_tracks_packed_weights_and_tactics_in_forward_defaults(changed):
    stamp = implementation().execution_stamp
    model = ModelMetadata()
    packed = SimpleNamespace(
        packed_weight=TensorMetadata(), weight_scale=TensorMetadata(), tactic=2
    )

    def forward(values, q=packed):
        return values, q

    model.forward = forward
    before = stamp(model)
    if changed == "tactic":
        packed.tactic += 1
    else:
        getattr(packed, changed)._version += 1
    assert stamp(model) != before


def test_execution_stamp_ignores_projection_group_transient_cache_values():
    stamp = implementation().execution_stamp
    model = ModelMetadata()
    group = SimpleNamespace(
        linear=SimpleNamespace(packed_weight=TensorMetadata()),
        _input=None,
        _output=None,
    )

    def forward(values, g=group):
        return values, g

    model.forward = forward
    before = stamp(model)
    group._input, group._output = TensorMetadata(), TensorMetadata()
    assert stamp(model) == before


def test_capture_retains_packed_constants_from_default_closures_and_excludes_transients():
    retain = getattr(implementation(), "captured_tensors", None)
    assert callable(retain), "Packed constant retention helper is missing"
    model = ModelMetadata()
    packed = SimpleNamespace(
        packed_weight=TensorMetadata(), weight_scale=TensorMetadata(), bias=None
    )
    group = SimpleNamespace(
        linear=packed, _input=TensorMetadata(), _output=TensorMetadata()
    )

    def forward(values, g=group):
        return values, g

    model.forward = forward
    actual = {id(value) for value in retain(model)}
    assert actual == {
        id(model.weight),
        id(model.buffer),
        id(packed.packed_weight),
        id(packed.weight_scale),
    }


def test_dropping_owned_binding_evicts_all_graphs_before_same_key_replacement():
    cls = implementation().PolicyStaticS2
    scope = cls.__new__(cls)
    scope.metadata_keys = ("input_ids", "attention_mask", "image_grid_thw")
    binding = {key: TensorMetadata() for key in scope.metadata_keys}
    key, other = ("same CPU contents",), ("other contents",)
    bucket = implementation().graph_bucket(key, TensorMetadata((16, 24)))
    changed_layout = implementation().graph_bucket(key, TensorMetadata((16, 48)))
    other_bucket = implementation().graph_bucket(other, TensorMetadata((16, 24)))
    scope.graphs = OrderedDict.fromkeys(
        (bucket, changed_layout, other_bucket), object()
    )
    scope._cpu_metadata = {key: object(), other: object()}
    scope._resident_keys = {
        tuple(id(binding[name]) for name in scope.metadata_keys): key
    }
    scope.core = SimpleNamespace(registrations={id(binding): object()})
    scope.counts = Counter()
    scope._drop_binding(key, binding)
    assert list(scope.graphs) == [other_bucket]
    assert scope.counts["graph_evictions"] == 2
    assert not scope._resident_keys and not scope.core.registrations


@pytest.mark.parametrize("dtype,device", [("fp32", "cuda"), ("bf16", "cpu")])
def test_direct_static_admission_rejects_non_bf16_or_non_cuda_metadata(dtype, device):
    check = getattr(implementation(), "check_policy_parameter", None)
    assert callable(check), "Policy-level parameter admission is missing"
    parameter = SimpleNamespace(
        dtype=dtype, device=SimpleNamespace(type=device), requires_grad=True
    )
    with pytest.raises(ValueError, match="CUDA BF16"):
        check(parameter, "bf16")


def test_direct_static_admission_accepts_eval_bf16_with_original_grad_flags():
    check = getattr(implementation(), "check_policy_parameter", None)
    assert callable(check), "Policy-level parameter admission is missing"
    parameter = SimpleNamespace(
        dtype="bf16", device=SimpleNamespace(type="cuda"), requires_grad=True
    )
    check(parameter, "bf16")
    assert parameter.requires_grad


def compiled_guard(model):
    cls = getattr(implementation(), "CompiledExecutionGuard", None)
    assert callable(cls), "Cold-compiled execution guard is missing"
    return cls(model)


def test_compiled_guard_hot_checks_never_repeat_module_or_recursive_discovery(
    monkeypatch,
):
    model = ModelMetadata()
    guard = compiled_guard(model)

    def forbidden(*args, **kwargs):
        pytest.fail("Hot guards must use the cold getter plan")

    for name in ("named_modules", "named_parameters", "named_buffers"):
        monkeypatch.setattr(model, name, forbidden)
    monkeypatch.setattr(implementation(), "_captured_stamp", forbidden)
    for _ in range(5):
        assert guard.unchanged()
    model.weight._version += 1
    assert not guard.unchanged()


@pytest.mark.parametrize(
    "changed",
    [
        "weight_version",
        "weight_replace",
        "buffer",
        "layout",
        "backend",
        "forward",
        "hook",
        "training",
    ],
)
def test_compiled_guard_preserves_registered_state_mutation_coverage(changed):
    model = ModelMetadata()
    guard = compiled_guard(model)
    if changed == "weight_version":
        model.weight._version += 1
    elif changed == "weight_replace":
        model.weight = TensorMetadata()
    elif changed == "buffer":
        model.buffer._version += 1
    elif changed == "layout":
        model.weight._stride = (1, 8)
    elif changed == "backend":
        model.config._attn_implementation = "eager"
    elif changed == "forward":
        model.forward = lambda values: (values, "new")
    elif changed == "hook":
        model._forward_hooks[5] = lambda *_args: None
    else:
        model.training = True
    assert not guard.unchanged()
    assert compiled_guard(model).unchanged()


@pytest.mark.parametrize("changed", ["packed_weight", "tactic", "default", "global"])
def test_compiled_guard_preserves_packed_closure_and_global_mutation_coverage(
    changed, monkeypatch
):
    model = ModelMetadata()
    packed = SimpleNamespace(
        packed_weight=TensorMetadata(), weight_scale=TensorMetadata(), tactic=2
    )

    def forward(values, q=packed):
        return apply_rotary_pos_emb(values), q

    model.forward = forward
    guard = compiled_guard(model)
    if changed == "packed_weight":
        packed.packed_weight._version += 1
    elif changed == "tactic":
        packed.tactic += 1
    elif changed == "default":
        forward.__defaults__ = (SimpleNamespace(packed_weight=TensorMetadata()),)
    else:
        monkeypatch.setitem(
            forward.__globals__, "apply_rotary_pos_emb", lambda values: (values, "new")
        )
    assert not guard.unchanged()
    assert compiled_guard(model).unchanged()


def test_compiled_guard_tracks_module_registry_topology_and_ignores_transient_projection_cache():
    model = ModelMetadata()
    model._modules = {"projection": SimpleNamespace()}
    group = SimpleNamespace(
        linear=SimpleNamespace(packed_weight=TensorMetadata()),
        _input=None,
        _output=None,
    )

    def forward(values, g=group):
        return values, g

    model.forward = forward
    guard = compiled_guard(model)
    group._input, group._output = TensorMetadata(), TensorMetadata()
    assert guard.unchanged()
    model._modules["projection"] = SimpleNamespace()
    assert not guard.unchanged()


def test_policy_hot_guard_uses_compiled_checks_and_rebuilds_epoch_on_fresh_preparation(
    monkeypatch,
):
    module = implementation()
    scope = module.PolicyStaticS2.__new__(module.PolicyStaticS2)
    model = ModelMetadata()
    cuda = SimpleNamespace(
        flash_sdp_enabled=lambda: True,
        mem_efficient_sdp_enabled=lambda: True,
        math_sdp_enabled=lambda: True,
        cudnn_sdp_enabled=lambda: True,
        matmul=SimpleNamespace(allow_tf32=False),
    )
    scope.torch = SimpleNamespace(
        backends=SimpleNamespace(cuda=cuda, cudnn=SimpleNamespace(allow_tf32=False))
    )
    scope.core = SimpleNamespace(
        inner=model, _model_stamp=lambda: ("metadata",), vision_rope=None
    )
    scope._execution_guard, scope._execution_epoch = None, 0
    initial = scope._execution_stamp(rebuild=True)

    def forbidden(*args, **kwargs):
        pytest.fail("Hot policy checks must not call the recursive reference stamp")

    monkeypatch.setattr(module, "execution_stamp", forbidden)
    monkeypatch.setattr(module, "_captured_stamp", forbidden)
    assert scope._execution_stamp() == initial
    model.weight._version += 1
    with pytest.raises(ValueError, match="execution changed"):
        scope._execution_stamp()
    rebuilt = scope._execution_stamp(rebuild=True)
    assert rebuilt != initial and scope._execution_epoch == 2
    assert scope.core.metadata_epoch == 1
    assert scope._execution_stamp() == rebuilt


def test_compiled_guard_rejects_owner_dictionary_replacement_before_using_old_getters():
    model = ModelMetadata()
    guard = compiled_guard(model)
    model.__dict__ = {**vars(model), "weight": TensorMetadata()}
    assert not guard.unchanged()
    assert compiled_guard(model).unchanged()


def test_hot_guard_batches_headers_and_containers_without_fingerprint_callbacks(
    monkeypatch,
):
    model = ModelMetadata()
    packed = SimpleNamespace(shape_tactics={16: 2}, packed_weight=TensorMetadata())

    def forward(values, q=packed):
        return values, q

    model.forward = forward
    guard = compiled_guard(model)
    module = importlib.import_module("coexecution.optimization.execution.guards")

    def forbidden(*args, **kwargs):
        pytest.fail("Hot comparisons must not dispatch a fingerprint per field")

    monkeypatch.setattr(module, "_identity", forbidden)
    assert guard.unchanged()
    packed.shape_tactics[16] = 3
    assert not guard.unchanged()


def test_cached_policy_topology_checks_s1_training_and_registry_changes_without_module_traversal(
    monkeypatch,
):
    module = importlib.import_module("coexecution.optimization.execution.guards")
    cls = getattr(module, "ModelTopologyGuard", None)
    assert callable(cls), "Cached policy topology admission is missing"
    model, s1 = ModelMetadata(), ModelMetadata()
    model._modules = {"action_head": s1}
    model.modules = lambda: iter((model, s1))
    guard = cls(model)

    def forbidden(*args, **kwargs):
        pytest.fail("Hot admission must not rediscover policy.modules")

    monkeypatch.setattr(model, "modules", forbidden)
    assert guard.unchanged() and not guard.training_changed()
    s1.training = True
    assert guard.training_changed()
    s1.training = False
    assert guard.unchanged()
    model._modules["action_head"] = ModelMetadata()
    assert not guard.unchanged()


def test_scalar_guard_rejects_tensor_replacement_without_equality_or_boolean_reads():
    class DeviceSentinel:
        def __eq__(self, other):
            pytest.fail("A host guard must not compare device tensor values")

        def __bool__(self):
            raise AssertionError("A host guard must not read a device tensor boolean")

    model = ModelMetadata()
    packed = SimpleNamespace(tactic=2)

    def forward(values, q=packed):
        return values, q

    model.forward = forward
    guard = compiled_guard(model)
    packed.tactic = DeviceSentinel()
    assert not guard.unchanged()
    topology = importlib.import_module(
        "coexecution.optimization.execution.guards"
    ).ModelTopologyGuard
    model.modules = lambda: iter([model])
    admission = topology(model)
    model.training = DeviceSentinel()
    assert admission.training_changed()


def test_packed_buffer_registry_replacement_invalidates_direct_registry_getters():
    class Packed:
        def __init__(self):
            self._buffers = {
                "packed_weight": TensorMetadata(),
                "weight_scale": TensorMetadata(),
            }

        def __getattr__(self, name):
            try:
                return self._buffers[name]
            except KeyError as error:
                raise AttributeError(name) from error

    model, packed = ModelMetadata(), Packed()

    def forward(values, q=packed):
        return values, q

    model.forward = forward
    guard = compiled_guard(model)
    packed._buffers = {
        "packed_weight": TensorMetadata(),
        "weight_scale": TensorMetadata(),
    }
    assert not guard.unchanged()


@pytest.mark.parametrize("mutation", ["defaults", "closure"])
def test_original_cold_mask_helper_changes_invalidate_compiled_guard_and_derived_epoch(
    mutation,
):
    module = implementation()
    scope = module.PolicyStaticS2.__new__(module.PolicyStaticS2)
    offset = [0]

    def helper(value=None, amount=0):
        return value, amount, offset[0]

    cuda = SimpleNamespace(
        flash_sdp_enabled=lambda: True,
        mem_efficient_sdp_enabled=lambda: True,
        math_sdp_enabled=lambda: True,
        cudnn_sdp_enabled=lambda: True,
        matmul=SimpleNamespace(allow_tf32=False),
    )
    scope.torch = SimpleNamespace(
        backends=SimpleNamespace(cuda=cuda, cudnn=SimpleNamespace(allow_tf32=False))
    )
    scope.core = SimpleNamespace(
        inner=ModelMetadata(),
        _model_stamp=lambda: ("metadata",),
        vision_rope=None,
        _causal_mask_original=helper,
    )
    scope._execution_guard, scope._execution_epoch = None, 0
    initial = scope._execution_stamp(rebuild=True)
    if mutation == "defaults":
        helper.__defaults__ = (None, 1)
    else:
        offset[0] = 1
    with pytest.raises(ValueError, match="execution changed"):
        scope._execution_stamp()
    assert scope._execution_stamp(rebuild=True) != initial
    assert scope.core.metadata_epoch == 1


@pytest.mark.parametrize("field", ["_causal_mask_original", "vision_rope"])
@pytest.mark.parametrize("mutation", ["defaults", "closure"])
def test_replaced_helper_repreparation_tracks_replacement_defaults_and_closure(
    field, mutation
):
    module = implementation()
    scope = module.PolicyStaticS2.__new__(module.PolicyStaticS2)
    offset = [0]

    def original(value=None, amount=0):
        return value, amount

    def replacement(value=None, amount=0):
        return value, amount, offset[0]

    cuda = SimpleNamespace(
        flash_sdp_enabled=lambda: True,
        mem_efficient_sdp_enabled=lambda: True,
        math_sdp_enabled=lambda: True,
        cudnn_sdp_enabled=lambda: True,
        matmul=SimpleNamespace(allow_tf32=False),
    )
    scope.torch = SimpleNamespace(
        backends=SimpleNamespace(cuda=cuda, cudnn=SimpleNamespace(allow_tf32=False))
    )
    scope.core = SimpleNamespace(
        inner=ModelMetadata(),
        _model_stamp=lambda: ("metadata",),
        vision_rope=None,
        _causal_mask_original=original,
    )
    if field == "vision_rope":
        scope.core.vision_rope = original
    scope._execution_guard, scope._execution_epoch = None, 0
    scope._execution_stamp(rebuild=True)
    setattr(scope.core, field, replacement)
    scope._execution_stamp(rebuild=True)
    if mutation == "defaults":
        replacement.__defaults__ = (None, 1)
    else:
        offset[0] = 1
    with pytest.raises(ValueError, match="execution changed"):
        scope._execution_stamp()
    scope._execution_stamp(rebuild=True)
    assert scope._execution_epoch == 3 and scope.core.metadata_epoch == 2


def test_inference_tensor_guard_uses_cheap_flag_without_accessing_unavailable_version():
    class InferenceTensorMetadata(TensorMetadata):
        @property
        def _version(self):
            pytest.fail(
                "Inference tensor version checks must not raise on every replay"
            )

        @_version.setter
        def _version(self, value):
            pass

        def is_inference(self):
            return True

    model = ModelMetadata()
    model.weight = InferenceTensorMetadata()
    assert compiled_guard(model).unchanged()
    assert implementation()._tensor_stamp(model.weight)[1] is None


def test_graph_retained_constant_discovery_is_cold_and_weights_remain_owned(
    monkeypatch,
):
    module = implementation()
    scope = module.PolicyStaticS2.__new__(module.PolicyStaticS2)
    model = ModelMetadata()
    scope.core = SimpleNamespace(
        inner=model,
        forward_inputs=lambda binding, pixels: {"backbone_features": pixels},
    )
    scope.graphs, scope.counts, scope.max_entries = OrderedDict(), Counter(), 32
    scope.BatchFeature = dict
    retained = []

    class MetadataGraph:
        def __init__(self, function, *, retained_tensors):
            self.function, self.retained_tensors = function, retained_tensors
            self.captures = self.replays = 0

        def __call__(self, pixels):
            self.captures = 1
            self.replays += 1
            retained.append(self.retained_tensors())
            return self.function(pixels)

    scope._graph_class = MetadataGraph
    binding = {"input_ids": TensorMetadata()}
    metadata = SimpleNamespace(
        tensors=lambda: (), attention_mask=TensorMetadata(), image_mask=TensorMetadata()
    )
    pixels = TensorMetadata((16, 24))
    scope._graph_forward(("metadata",), binding, pixels, metadata)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Retained tensor discovery must be cold, even if the helper calls its callback on replay"
        )

    monkeypatch.setattr(module, "captured_tensors", forbidden)
    scope._graph_forward(("metadata",), binding, pixels, metadata)
    assert retained[0] is retained[1]
    assert model.weight in retained[0] and model.buffer in retained[0]
    guard = compiled_guard(model)
    model.weight._version += 1
    assert not guard.unchanged()


@pytest.mark.parametrize("field", ["input_ids", "attention_mask", "image_grid_thw"])
def test_graph_bucket_includes_metadata_signature_and_exact_pixel_layout(field):
    key = implementation().graph_bucket
    metadata = (field, (1, 4, 4))
    pixels = TensorMetadata((16, 24))
    before = key(metadata, pixels)
    assert key(metadata, pixels) == before
    assert key((field, (1, 2, 8)), pixels) != before
    pixels._version += 1
    assert key(metadata, pixels) == before, "Pixel values must be refreshed, not cached"
    pixels._stride = (1, 16)
    assert key(metadata, pixels) != before
