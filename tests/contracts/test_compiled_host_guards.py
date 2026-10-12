"""Compilation wiring and cache contracts; native compilation is separately opt-in."""

import importlib
import importlib.util
import json
import multiprocessing
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from coexecution.optimization.execution.guards import HostGuardPlan, ModelTopologyGuard
from tests.contracts.test_policy_vlm import ModelMetadata

MODULE = "coexecution.optimization.execution.compiled_guards"


def implementation():
    assert importlib.util.find_spec(MODULE) is not None, (
        "Compiled guard backend missing"
    )
    return importlib.import_module(MODULE)


@pytest.fixture
def compiler(monkeypatch, tmp_path):
    module = implementation()
    builds = []
    source = tmp_path / "guards.py"
    source.write_bytes(module.SOURCE.read_bytes())
    monkeypatch.setattr(module, "SOURCE", source)
    monkeypatch.setattr(module, "_loaded", {})
    monkeypatch.setattr(
        module, "_build_settings", lambda: {"cython": "3.1.2", "abi": "test"}
    )

    def build(directory, name, source_bytes, settings):
        builds.append(name)
        assert source_bytes == source.read_bytes()
        binary = directory / (name + ".so")
        binary.write_bytes(b"native stand-in")
        return binary

    def load(name, path):
        # Independent Python functions stand in for compiled functions. No native
        # compiler or model framework runs in normal CI.
        namespace = {}
        exec(compile(source.read_bytes(), str(source), "exec"), namespace)  # noqa: S102 -- repository source only
        return SimpleNamespace(**namespace)

    monkeypatch.setattr(module, "_build", build)
    monkeypatch.setattr(module, "_import_extension", load)
    return SimpleNamespace(
        module=module, source=source, builds=builds, cache=tmp_path / "cache"
    )


def test_instance_binding_keeps_python_classes_and_other_sessions_unchanged(compiler):
    backend = compiler.module.load_compiled_guards(cache_dir=compiler.cache)
    original = HostGuardPlan.unchanged
    model = ModelMetadata()
    model.modules = lambda: iter((model,))
    plain = HostGuardPlan(model, methods=("forward",), fields=(), global_bindings=())
    native = HostGuardPlan(
        model, methods=("forward",), fields=(), global_bindings=(), backend=backend
    )
    topology = ModelTopologyGuard(model, backend=backend)
    assert HostGuardPlan.unchanged is original
    assert plain.unchanged.__func__ is original
    assert native.unchanged.__func__ is not original
    assert (
        native.unchanged() and topology.unchanged() and not topology.training_changed()
    )
    model.weight._version += 1
    assert not native.unchanged() and not plain.unchanged()
    model.training = True
    assert topology.training_changed()


def test_cache_identity_changes_with_source_and_build_settings(compiler, monkeypatch):
    module = compiler.module
    first = module.load_compiled_guards(cache_dir=compiler.cache)
    compiler.source.write_bytes(compiler.source.read_bytes() + b"\n# source revision\n")
    second = module.load_compiled_guards(cache_dir=compiler.cache)
    monkeypatch.setattr(
        module, "_build_settings", lambda: {"cython": "3.1.2", "abi": "different"}
    )
    third = module.load_compiled_guards(cache_dir=compiler.cache)
    assert len(compiler.builds) == 3
    assert len({item.evidence()["cache_key"] for item in (first, second, third)}) == 3
    assert first.evidence()["source_sha256"] != second.evidence()["source_sha256"]


def test_concurrent_load_builds_once_and_reports_verified_binary(compiler):
    barrier = threading.Barrier(4)

    def load(_):
        barrier.wait()
        return compiler.module.load_compiled_guards(cache_dir=compiler.cache)

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(load, range(4)))
    assert len(compiler.builds) == 1
    assert all(item is results[0] for item in results)
    evidence = results[0].evidence()
    assert evidence["backend"] == "cython"
    assert len(evidence["binary_sha256"]) == 64
    assert evidence["build_settings"] == {"cython": "3.1.2", "abi": "test"}
    published = list(compiler.cache.glob("*/manifest.json"))
    assert len(published) == 1
    assert (
        json.loads(published[0].read_text())["binary_sha256"]
        == evidence["binary_sha256"]
    )


def test_corrupt_cache_is_rebuilt_and_failures_do_not_publish(compiler, monkeypatch):
    backend = compiler.module.load_compiled_guards(cache_dir=compiler.cache)
    binary = (
        compiler.cache / backend.evidence()["cache_key"] / backend.evidence()["binary"]
    )
    binary.write_bytes(b"corruption")
    monkeypatch.setattr(compiler.module, "_loaded", {})
    compiler.module.load_compiled_guards(cache_dir=compiler.cache)
    assert len(compiler.builds) == 2
    compiler.source.write_bytes(compiler.source.read_bytes() + b"\n# fail\n")

    def fail(*args):
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(compiler.module, "_build", fail)
    with pytest.raises(RuntimeError, match="compiler unavailable"):
        compiler.module.load_compiled_guards(cache_dir=compiler.cache)
    assert len(list(compiler.cache.glob("*/manifest.json"))) == 1


def test_backend_import_has_no_framework_or_compiler_side_effects():
    import subprocess

    implementation()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; import {MODULE}; assert not any(name.startswith(('torch', 'Cython', 'setuptools', 'numpy')) for name in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_invalid_manifest_shape_is_rebuilt(compiler, monkeypatch):
    backend = compiler.module.load_compiled_guards(cache_dir=compiler.cache)
    manifest = compiler.cache / backend.evidence()["cache_key"] / "manifest.json"
    manifest.write_text("[]")
    monkeypatch.setattr(compiler.module, "_loaded", {})
    compiler.module.load_compiled_guards(cache_dir=compiler.cache)
    assert len(compiler.builds) == 2


def test_independent_processes_publish_one_complete_cache_entry(
    compiler, monkeypatch, tmp_path
):
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(3)
    builds = tmp_path / "builds"
    original = compiler.module._build

    def build(*args):
        with builds.open("a") as record:
            record.write("build\n")
        return original(*args)

    monkeypatch.setattr(compiler.module, "_build", build)

    def load():
        barrier.wait(timeout=5)
        backend = compiler.module.load_compiled_guards(cache_dir=compiler.cache)
        assert backend.evidence()["binary_sha256"]

    processes = [context.Process(target=load) for _ in range(3)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert builds.read_text() == "build\n"
    assert len(list(compiler.cache.glob("*/manifest.json"))) == 1


def test_vlm_execution_guard_rebuild_preserves_selected_backend(compiler):
    from coexecution.optimization.groot.vlm import PolicyStaticS2

    backend = compiler.module.load_compiled_guards(cache_dir=compiler.cache)
    model = ModelMetadata()
    scope = PolicyStaticS2.__new__(PolicyStaticS2)
    scope.guard_backend = backend
    scope.core = SimpleNamespace(inner=model, _model_stamp=lambda: (), vision_rope=None)
    scope._execution_guard, scope._execution_epoch = None, 0
    scope.torch = SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(
                flash_sdp_enabled=lambda: True,
                mem_efficient_sdp_enabled=lambda: True,
                math_sdp_enabled=lambda: True,
                cudnn_sdp_enabled=lambda: True,
                matmul=SimpleNamespace(allow_tf32=False),
            ),
            cudnn=SimpleNamespace(allow_tf32=False),
        )
    )
    scope._execution_stamp(rebuild=True)
    first = scope._execution_guard
    assert first.unchanged.__func__ is backend.module.HostGuardPlan.unchanged
    model._forward_hooks[1] = lambda *args: None
    with pytest.raises(ValueError, match="execution changed"):
        scope._execution_stamp()
    scope._execution_stamp(rebuild=True)
    assert scope._execution_guard is not first
    assert (
        scope._execution_guard.unchanged.__func__
        is backend.module.HostGuardPlan.unchanged
    )
