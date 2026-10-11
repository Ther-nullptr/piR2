"""Input separation and scope contracts using metadata-only GR00T interfaces."""

import importlib
import importlib.util
import subprocess
import sys
import threading
from collections import UserDict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREPARATION = "coexecution.optimization.execution.preparation"
BACKBONE_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
ACTION_KEYS = ("state", "embodiment_id", "action", "action_mask", "image_delay")


class BatchFeature(UserDict):
    """Only container semantics are needed; no tensor or model computations."""

    def __init__(self, data):
        super().__init__(data)


@pytest.fixture
def interface(monkeypatch):
    calls = []

    class Qwen3Backbone:
        training = False

        def prepare_input(self, batch):
            calls.append(("backbone", batch))
            return BatchFeature(data=batch)

        def forward(self, batch):
            return {key: batch[key] for key in BACKBONE_KEYS}

    class Gr00tN1d7ActionHead:
        training = False

        def prepare_input(self, batch):
            calls.append(("action", batch))
            return BatchFeature(data=batch)

        def get_action(self, backbone, action, options=None):
            return backbone, action, options

    class Gr00tN1d7:
        training = False

        def __init__(self):
            self.backbone = Qwen3Backbone()
            self.action_head = Gr00tN1d7ActionHead()
            self.parameter = SimpleNamespace(
                dtype="bf16", device=SimpleNamespace(type="cuda"), requires_grad=False
            )
            self.config = SimpleNamespace(streaming=False)

        def parameters(self):
            return iter([self.parameter])

        def modules(self):
            return iter([self, self.backbone, self.action_head])

        def prepare_input(self, batch):
            if "vlm_content" in batch:
                batch.update(batch.pop("vlm_content"))
            return (
                self.backbone.prepare_input(batch),
                self.action_head.prepare_input(batch),
            )

        def get_action(self, batch, options=None):
            backbone, action = self.prepare_input(batch)
            return self.action_head.get_action(backbone, action, options)

    for name, exports in (
        ("torch", {"bfloat16": "bf16"}),
        (
            "gr00t.model.gr00t_n1d7.gr00t_n1d7",
            {"Gr00tN1d7": Gr00tN1d7, "Gr00tN1d7ActionHead": Gr00tN1d7ActionHead},
        ),
        ("gr00t.model.modules.qwen3_backbone", {"Qwen3Backbone": Qwen3Backbone}),
        ("transformers.feature_extraction_utils", {"BatchFeature": BatchFeature}),
    ):
        module = ModuleType(name)
        vars(module).update(exports)
        monkeypatch.setitem(sys.modules, name, module)
    model = Gr00tN1d7()
    return SimpleNamespace(
        policy=SimpleNamespace(model=model),
        model=model,
        calls=calls,
        model_type=Gr00tN1d7,
        backbone_type=Qwen3Backbone,
    )


def scope(policy):
    assert importlib.util.find_spec(PREPARATION) is not None, (
        "InputPreparation module is missing"
    )
    return importlib.import_module(PREPARATION).InputPreparation(policy)


def inputs(*, optional=False):
    keys = (
        (*BACKBONE_KEYS, *ACTION_KEYS)
        if optional
        else (*BACKBONE_KEYS, "state", "embodiment_id")
    )
    return {**{key: object() for key in keys}, "unused_metadata": object()}


def test_preparation_import_is_metadata_only():
    assert importlib.util.find_spec(PREPARATION) is not None, (
        "InputPreparation module is missing"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import sys
from {PREPARATION} import InputPreparation
assert callable(InputPreparation)
assert not any(name.startswith(("torch", "transformers", "gr00t", "triton"))
               for name in sys.modules)
""",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("optional", [False, True])
def test_components_receive_disjoint_owned_inputs(interface, optional):
    batch = inputs(optional=optional)
    saved = dict(batch)
    with scope(interface.policy) as adapter:
        backbone, action = interface.model.prepare_input(batch)
        assert isinstance(backbone, BatchFeature) and isinstance(action, BatchFeature)
        assert tuple(backbone) == BACKBONE_KEYS
        assert tuple(action) == (ACTION_KEYS if optional else ACTION_KEYS[:2])
        assert not backbone.keys() & action.keys()
        assert all(value is saved[key] for key, value in backbone.items())
        assert all(value is saved[key] for key, value in action.items())
        assert all(received is not batch for _, received in interface.calls)
        assert adapter.evidence()["backbone_keys"] == list(BACKBONE_KEYS)
        assert adapter.evidence()["action_keys"] == list(ACTION_KEYS)
    assert batch == saved


def test_changed_observations_return_fresh_containers_and_fields(interface):
    first, second = inputs(), inputs()
    with scope(interface.policy):
        previous = interface.model.prepare_input(first)
        current = interface.model.prepare_input(second)
        assert all(old is not new for old, new in zip(previous, current))
        assert previous[0]["pixel_values"] is first["pixel_values"]
        assert current[0]["pixel_values"] is second["pixel_values"]
        assert previous[1]["state"] is first["state"]
        assert current[1]["state"] is second["state"]
        current[0]["pixel_values"] = object()
        current[1]["state"] = object()
    assert previous[0]["pixel_values"] is first["pixel_values"]
    assert previous[1]["state"] is first["state"]


def test_captured_original_preprocessing_and_options_are_preserved(interface):
    normalized = object()
    original = interface.model.backbone.prepare_input

    def prepare(batch):
        prepared = original(batch)
        prepared["pixel_values"] = normalized
        return prepared

    interface.model.backbone.prepare_input = prepare
    options = {"rtc_overlap_steps": 2, "image_delay": 3, "force_nonstreaming": True}
    batch = inputs(optional=True)
    pixel = batch["pixel_values"]
    with scope(interface.policy):
        backbone, action, actual_options = interface.model.get_action(batch, options)
        assert backbone["pixel_values"] is normalized
        assert action["action"] is batch["action"]
        assert actual_options is options
    assert interface.model.backbone.prepare_input is prepare
    assert batch["pixel_values"] is pixel


def test_original_conditional_collation_does_not_mutate_caller(interface):
    batch = inputs()
    content = {key: batch.pop(key) for key in BACKBONE_KEYS}
    batch["vlm_content"] = content
    saved = dict(batch)
    with scope(interface.policy):
        backbone, action = interface.model.prepare_input(batch)
        assert backbone == content
        assert tuple(action) == ACTION_KEYS[:2]
    assert batch == saved


@pytest.mark.parametrize("has_override", [False, True])
def test_attribute_presence_and_originals_are_restored(interface, has_override):
    model = interface.model
    targets = (model, model.backbone, model.action_head)
    if has_override:
        for target in targets:
            target.prepare_input = target.prepare_input
    saved = [
        (target.prepare_input, "prepare_input" in vars(target)) for target in targets
    ]
    adapter = scope(interface.policy)
    with pytest.raises(RuntimeError, match="body"), adapter:
        model.prepare_input(inputs())
        raise RuntimeError("body failed")
    for target, (original, had) in zip(targets, saved):
        assert target.prepare_input == original
        assert ("prepare_input" in vars(target)) == had
    adapter.close()


@pytest.mark.parametrize("change", ["model", "backbone", "action_head"])
def test_replacing_a_scoped_model_or_component_is_rejected(interface, change):
    model = interface.model
    with scope(interface.policy):
        if change == "model":
            interface.policy.model = interface.model_type()
        else:
            setattr(model, change, object())
        with pytest.raises(RuntimeError, match="replaced"):
            model.prepare_input(inputs())


@pytest.mark.parametrize("target", ["model", "backbone", "action_head"])
def test_unsupported_classes_are_rejected_before_any_patch(interface, target):
    model = interface.model
    if target == "model":
        interface.policy.model = object()
    else:
        setattr(model, target, object())
    with pytest.raises(ValueError, match="Unsupported"), scope(interface.policy):
        pytest.fail("Unsupported interfaces must not enter the scope")
    assert "prepare_input" not in vars(model)


@pytest.mark.parametrize("target", ["model", "backbone", "action_head"])
def test_training_is_rejected_at_entry_and_during_preparation(interface, target):
    module = interface.model if target == "model" else getattr(interface.model, target)
    module.training = True
    with pytest.raises(ValueError, match="eval"), scope(interface.policy):
        pytest.fail("Training must not enter the scope")
    module.training = False
    with scope(interface.policy):
        module.training = True
        with pytest.raises((ValueError, RuntimeError), match="eval"):
            interface.model.prepare_input(inputs())
        module.training = False
        interface.model.prepare_input(inputs())


@pytest.mark.parametrize(
    "attribute,value,message",
    [
        ("dtype", "fp32", "BF16"),
        ("device", SimpleNamespace(type="cpu"), "CUDA"),
    ],
)
def test_direct_scope_requires_cuda_bf16(interface, attribute, value, message):
    setattr(interface.model.parameter, attribute, value)
    with pytest.raises(ValueError, match=message), scope(interface.policy):
        pytest.fail("Direct scopes must enforce model admission")


def test_eval_policy_admission_preserves_original_trainable_parameter_flags(interface):
    interface.model.parameter.requires_grad = True
    with scope(interface.policy):
        interface.model.prepare_input(inputs())
    assert interface.model.parameter.requires_grad


@pytest.mark.parametrize("target,index", [("backbone", 0), ("action_head", 1)])
@pytest.mark.parametrize("hook_kind", ["forward_hooks", "forward_pre_hooks"])
def test_unknown_component_hooks_keep_full_inputs_and_record_fallback(
    interface, target, index, hook_kind
):
    module = getattr(interface.model, target)
    batch = inputs()
    with scope(interface.policy) as preparation:
        # Hooks can arrive after preparation installation, as in the policy backend.
        setattr(module, "_" + hook_kind, {1: object()})
        prepared = interface.model.prepare_input(batch)
        assert tuple(prepared[index]) == tuple(batch)
        assert prepared[index] is not batch
        assert all(value is batch[key] for key, value in prepared[index].items())
        assert preparation.evidence()["fallbacks"][target] == [hook_kind]
        getattr(module, "_" + hook_kind).clear()
        separated = interface.model.prepare_input(batch)
        assert tuple(separated[0]) == BACKBONE_KEYS
        assert tuple(separated[1]) == ACTION_KEYS[:2]


@pytest.mark.parametrize("hook_kind", ["forward_hooks", "forward_pre_hooks"])
@pytest.mark.parametrize("before_entry", [False, True])
def test_global_hooks_preserve_full_component_inputs_and_record_fallback(
    interface, monkeypatch, hook_kind, before_entry
):
    registry = ModuleType("torch.nn.modules.module")
    hooks = {}
    setattr(registry, "_global_" + hook_kind, hooks)
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    if before_entry:
        hooks[1] = object()
    batch = inputs()
    with scope(interface.policy) as preparation:
        if not before_entry:
            hooks[1] = object()
        prepared = interface.model.prepare_input(batch)
        assert all(tuple(part) == tuple(batch) for part in prepared)
        assert all(
            value is batch[key] for part in prepared for key, value in part.items()
        )
        assert preparation.evidence()["fallbacks"] == {
            "backbone": ["global_" + hook_kind],
            "action_head": ["global_" + hook_kind],
        }
        prepared[0]["state"] = object()
        assert prepared[1]["state"] is batch["state"]
        hooks.clear()
        separated = interface.model.prepare_input(batch)
        assert tuple(separated[0]) == BACKBONE_KEYS
        assert tuple(separated[1]) == ACTION_KEYS[:2]


def test_changed_consumption_or_preparation_methods_are_rejected(interface):
    original = interface.backbone_type.forward
    with scope(interface.policy):
        interface.backbone_type.forward = lambda self, batch: batch["extra_visual"]
        try:
            with pytest.raises(RuntimeError, match="changed"):
                interface.model.prepare_input(inputs())
        finally:
            interface.backbone_type.forward = original

        def replacement(batch):
            return BatchFeature(data=batch)

        interface.model.action_head.prepare_input = replacement
        with pytest.raises(RuntimeError, match="changed"):
            interface.model.prepare_input(inputs())


@pytest.mark.parametrize("target", ["model", "backbone", "action_head"])
def test_unknown_consumption_overrides_are_rejected_at_entry(interface, target):
    module = interface.model if target == "model" else getattr(interface.model, target)
    name = "get_action" if target != "backbone" else "forward"
    setattr(module, name, lambda *args, **kwargs: None)
    with (
        pytest.raises(ValueError, match="Unsupported.*consumption"),
        scope(interface.policy),
    ):
        pytest.fail("Unknown consumers must not be admitted for field separation")


def test_session_close_during_preparation_keeps_the_boundary_installed(
    interface, monkeypatch
):
    from coexecution.optimization.config import OptimizationConfig
    from coexecution.optimization.session import GrootOptimizations

    monkeypatch.setattr(
        "coexecution.optimization.session.preflight_backend", lambda *args: {}
    )
    original = interface.model.backbone.prepare_input
    session = GrootOptimizations(
        interface.policy, OptimizationConfig(input_preparation=True)
    )

    def prepare(batch):
        with pytest.raises(RuntimeError, match="active"):
            session.close()
        assert not session.boundary.closed
        return original(batch)

    interface.model.backbone.prepare_input = prepare
    with session:
        interface.model.prepare_input(inputs())
        assert not session.boundary.closed
    assert interface.model.backbone.prepare_input is prepare


def test_unknown_new_prepared_fields_are_rejected_and_recovery_is_possible(interface):
    original = interface.model.backbone.prepare_input
    add_unknown = [True]

    def prepare(batch):
        prepared = original(batch)
        if add_unknown[0]:
            prepared["additional_visual_input"] = object()
        return prepared

    interface.model.backbone.prepare_input = prepare
    with scope(interface.policy):
        with pytest.raises(ValueError, match="Unsupported prepared"):
            interface.model.prepare_input(inputs())
        add_unknown[0] = False
        assert tuple(interface.model.prepare_input(inputs())[0]) == BACKBONE_KEYS


def test_failed_original_preparation_leaves_scope_usable(interface):
    original = interface.model.backbone.prepare_input
    fail = [True]

    def prepare(batch):
        if fail[0]:
            raise RuntimeError("prepare failure")
        return original(batch)

    interface.model.backbone.prepare_input = prepare
    with scope(interface.policy):
        with pytest.raises(RuntimeError, match="prepare failure"):
            interface.model.prepare_input(inputs())
        fail[0] = False
        assert tuple(interface.model.prepare_input(inputs())[0]) == BACKBONE_KEYS
    assert interface.model.backbone.prepare_input is prepare


def test_concurrent_preparation_and_close_during_preparation_are_rejected(interface):
    original = interface.model.backbone.prepare_input
    started, release = threading.Event(), threading.Event()
    failures = []

    def prepare(batch):
        started.set()
        assert release.wait(5)
        return original(batch)

    interface.model.backbone.prepare_input = prepare
    adapter = scope(interface.policy)

    def submit():
        try:
            interface.model.prepare_input(inputs())
        # Retain thread failures for the assertion in the parent test.
        except BaseException as error:  # noqa: BLE001
            failures.append(error)

    with adapter:
        worker = threading.Thread(target=submit)
        worker.start()
        try:
            assert started.wait(5)
            with pytest.raises(RuntimeError, match="serial"):
                interface.model.prepare_input(inputs())
            with pytest.raises(RuntimeError, match="active"):
                adapter.close()
        finally:
            release.set()
            worker.join(5)
        assert not worker.is_alive()
        assert not failures
        interface.model.prepare_input(inputs())
    assert interface.model.backbone.prepare_input is prepare


def test_partial_install_failure_restores_already_installed_methods(interface):
    original = interface.model.action_head

    class RefusePatch(type(original)):
        def __setattr__(self, name, value):
            if name == "prepare_input":
                raise RuntimeError("patch refused")
            super().__setattr__(name, value)

    original.__class__ = RefusePatch
    # Admission is tested separately; this sentinel retains the exact admitted type.
    module = sys.modules["gr00t.model.gr00t_n1d7.gr00t_n1d7"]
    module.Gr00tN1d7ActionHead = RefusePatch
    adapter = scope(interface.policy)
    with pytest.raises(RuntimeError, match="patch refused"), adapter:
        pytest.fail("Failed installation cannot enter its body")
    assert "prepare_input" not in vars(interface.model)
    assert "prepare_input" not in vars(interface.model.backbone)
    assert "prepare_input" not in vars(original)
