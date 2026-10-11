"""Scoped construction contracts without importing or executing a model."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from coexecution.libero.loading import task_checkpoint_backbone


@pytest.fixture(params=[False, True], ids=["inherited", "owned"])
def loaders(monkeypatch, request):
    class Base:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return (cls, name, kwargs)

    class Model(Base):
        _from_config = Mock(return_value=object())

    if request.param:
        Model.from_pretrained = Base.__dict__["from_pretrained"]
    config = Mock(return_value=object())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            Qwen3VLConfig=SimpleNamespace(from_pretrained=config),
            Qwen3VLForConditionalGeneration=Model,
        ),
    )
    return Model, config


@pytest.mark.parametrize("dtype_key", ["torch_dtype", "dtype"])
def test_task_checkpoint_preserves_loading_and_model_options(loaders, dtype_key):
    model, config = loaders
    original = model.__dict__.get("from_pretrained")
    loading = {
        "local_files_only": True,
        "cache_dir": "cache",
        "revision": "pinned",
        "trust_remote_code": False,
        "token": "test-token",
    }
    model_options = {dtype_key: object(), "attn_implementation": "flash_attention_2"}
    with task_checkpoint_backbone():
        result = model.from_pretrained("backbone-metadata", **loading, **model_options)
    config.assert_called_once_with("backbone-metadata", **loading)
    model._from_config.assert_called_once_with(config.return_value, **model_options)
    assert result is model._from_config.return_value
    assert model.__dict__.get("from_pretrained") is original
    assert model.from_pretrained("weights") == (model, "weights", {})


@pytest.mark.parametrize("failure", ["config", "model", "checkpoint"])
def test_task_checkpoint_restores_descriptor_on_failure(loaders, failure):
    model, config = loaders
    original = model.__dict__.get("from_pretrained")
    if failure == "config":
        config.side_effect = RuntimeError("config")
    elif failure == "model":
        model._from_config.side_effect = RuntimeError("model")
    with pytest.raises(RuntimeError, match=failure), task_checkpoint_backbone():
        model.from_pretrained("metadata")
        raise RuntimeError("checkpoint")
    assert model.__dict__.get("from_pretrained") is original


def test_task_checkpoint_leaves_unspecified_model_options_unset(loaders):
    model, config = loaders
    with task_checkpoint_backbone():
        model.from_pretrained("metadata", local_files_only=True)
    model._from_config.assert_called_once_with(config.return_value)


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
        "constructor",
    ],
)
def test_strict_task_loading_restores_loader_and_cuda_context(
    tmp_path, monkeypatch, failure
):
    from contextlib import contextmanager

    from coexecution.libero import loading, service

    active = []
    info = {failure: ["test entry"]} if failure not in (None, "constructor") else {}
    original = Mock(return_value=(object(), info))
    policy_module = SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=original))

    @contextmanager
    def checkpoint_scope():
        active.append("checkpoint")
        try:
            yield
        finally:
            assert active.pop() == "checkpoint"

    @contextmanager
    def device_scope(device):
        active.append(device)
        try:
            yield
        finally:
            assert active.pop() == device

    core = object()

    def construct(embodiment, checkpoint, *, device):
        assert (embodiment, checkpoint, device) == (
            "LIBERO_PANDA",
            "checkpoint",
            "cuda:1",
        )
        assert active == ["checkpoint", "cuda:1"]
        policy_module.AutoModel.from_pretrained(checkpoint, strict_option="preserved")
        if failure == "constructor":
            raise RuntimeError("constructor")
        return core

    def synchronize(device):
        assert active == ["checkpoint", device]

    synchronized = Mock(side_effect=synchronize)
    monkeypatch.setattr(loading, "task_checkpoint_backbone", checkpoint_scope)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(device=device_scope, synchronize=synchronized)
        ),
    )
    monkeypatch.setitem(
        sys.modules, "gr00t.policy", SimpleNamespace(gr00t_policy=policy_module)
    )
    monkeypatch.setitem(
        sys.modules,
        "gr00t.policy.decoupled_policy",
        SimpleNamespace(DecoupledGr00tPolicy=construct),
    )
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            service.load_core("checkpoint", tmp_path, "cuda:1")
        synchronized.assert_not_called()
    else:
        assert service.load_core("checkpoint", tmp_path, "cuda:1") is core
        synchronized.assert_called_once_with("cuda:1")
    assert active == []
    assert policy_module.AutoModel.from_pretrained is original
    original.assert_called_once_with(
        "checkpoint", strict_option="preserved", output_loading_info=True
    )
    assert (tmp_path / "weight-loading.json").read_text() == service.json.dumps(
        info, indent=2
    ) + "\n"
