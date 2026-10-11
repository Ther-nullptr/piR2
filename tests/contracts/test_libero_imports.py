"""LIBERO runtime import, CLI and provenance contracts without model execution."""

import ast
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Fail before any import can initialize a model, CUDA runtime or simulator.
IMPORT_GUARD = """
import sys

class RejectRuntimeDependencies:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'transformers', 'gr00t', 'triton', 'robotics_kernels',
            'libero', 'mujoco', 'robosuite',
        }:
            raise AssertionError('Unexpected runtime import: ' + fullname)

sys.meta_path.insert(0, RejectRuntimeDependencies())
"""


def run_import_contract(program, tmp_path, *args):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_GUARD + program, str(ROOT), *args],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_canonical_runtime_imports_do_not_require_scripts_on_sys_path(tmp_path):
    run_import_contract(
        """
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from coexecution.libero import backend, loading, service
from scripts import libero_experiment_utils, libero_protocol_scheduler

assert backend.extra_buffer_shift is libero_protocol_scheduler.extra_buffer_shift
assert backend.project_commands is libero_protocol_scheduler.project_commands
assert service.Backend is backend.Backend
assert str(Path(sys.argv[1]) / 'scripts') not in sys.path
assert callable(libero_experiment_utils.checkpoint_identity)
assert callable(loading.task_checkpoint_backbone)
""",
        tmp_path,
    )


@pytest.mark.parametrize("legacy_mode", ["script_directory", "package"])
def test_legacy_imports_keep_export_identity_from_an_unrelated_cwd(
    tmp_path, legacy_mode
):
    run_import_contract(
        """
import importlib
from pathlib import Path
root = Path(sys.argv[1])
mode = sys.argv[2]
sys.path.insert(0, str(root / 'scripts' if mode == 'script_directory' else root))
prefix = '' if mode == 'script_directory' else 'scripts.'
for legacy, canonical, expected in (
    ('libero_model_loading', 'loading', ('contextmanager', 'task_checkpoint_backbone')),
    ('libero_inference_backend', 'backend', (
        'KEYS', 'Backend', 'endpoint_layout', 'extra_buffer_shift',
        'hashlib', 'json', 'np', 'project_commands', 're', 'time',
    )),
    ('serve_libero_protocol', 'service', (
        'Backend', 'GrootOptimizations', 'LOGGER', 'Path',
        'add_optimization_arguments', 'argparse', 'endpoint_layout', 'json',
        'load_core', 'logging', 'main', 'optimization_config', 'os', 'serve',
        'supervise_workers', 'threading',
    )),
):
    old = importlib.import_module(prefix + legacy)
    new = importlib.import_module('coexecution.libero.' + canonical)
    assert set(old.__all__) == set(expected)
    for name in expected:
        assert getattr(old, name) is getattr(new, name), (legacy, name)
""",
        tmp_path,
        legacy_mode,
    )


def test_legacy_server_help_works_outside_repository_without_gpu_imports(tmp_path):
    help_text = run_import_contract(
        """
from pathlib import Path
import runpy
script = Path(sys.argv[1]) / 'scripts/serve_libero_protocol.py'
sys.argv = [str(script), '--help']
try:
    runpy.run_path(str(script), run_name='__main__')
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError('Expected argparse help exit')
""",
        tmp_path,
    )
    for flag in (
        "--checkpoint",
        "--variant",
        "--vlm-device",
        "--operator-fusion",
        "--inference-precision",
        "--dit-cuda-graph",
    ):
        assert flag in help_text


def test_evaluation_fingerprints_include_canonical_runtime_implementations(tmp_path):
    from coexecution.libero import backend, loading, service

    source = ROOT / "scripts/evaluate_libero_protocol.py"
    module = ast.parse(source.read_text())
    main = next(node for node in module.body if getattr(node, "name", None) == "main")
    files_assignment = next(
        node
        for node in main.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "implementation_files"
            for target in node.targets
        )
    )
    recorded_files = ast.literal_eval(files_assignment.value)
    runtime_files = [
        Path(item.__file__).relative_to(ROOT).as_posix()
        for item in (loading, backend, service)
    ]
    assert set(runtime_files) <= set(recorded_files)
    assert {
        "scripts/serve_libero_protocol.py",
        "scripts/libero_inference_backend.py",
        "scripts/libero_model_loading.py",
    } <= set(recorded_files)

    hashes_assignment = next(
        node
        for node in main.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "implementation_sha256"
            for target in node.targets
        )
    )
    statement = ast.Module(body=[hashes_assignment], type_ignores=[])
    program = compile(statement, str(source), "exec")
    for name in runtime_files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    context = {
        "config": {},
        "hashlib": hashlib,
        "root": tmp_path,
        "implementation_files": runtime_files,
    }
    exec(program, context)  # noqa: S102 -- trusted local hash statement only
    before = dict(context["config"]["implementation_sha256"])
    assert before == {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in runtime_files
    }
    # A canonical implementation change must be visible even with an unchanged shim.
    changed = runtime_files[1]
    with (tmp_path / changed).open("ab") as output:
        output.write(b"\n# provenance contract change\n")
    exec(program, context)  # noqa: S102 -- trusted local hash statement only
    after = context["config"]["implementation_sha256"]
    assert {name for name in runtime_files if before[name] != after[name]} == {changed}
