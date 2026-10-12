"""Vision admission CLI/configuration provenance without loading model runtimes."""

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def evaluator_main():
    tree = ast.parse((ROOT / "scripts/evaluate_libero_protocol.py").read_text())
    return next(node for node in tree.body if getattr(node, "name", None) == "main")


def assignment(main, target):
    return next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(ast.unparse(name) == target for name in node.targets)
    )


def execute_assignment(node, context):
    program = compile(
        ast.Module(body=[node], type_ignores=[]), "config-contract", "exec"
    )
    exec(program, context)  # noqa: S102 -- evaluate trusted local metadata expressions


def test_requested_rate_and_deployment_contract_are_saved_and_bind_resume(tmp_path):
    from evaluate_libero_protocol import build_parser, save_run_configuration

    args = build_parser().parse_args(
        ["--protocol", "deployment", "--variant", "flow", "--output", str(tmp_path)]
    )
    args.vision_max_hz = 15
    main = evaluator_main()
    context = {"args": args}
    execute_assignment(assignment(main, "config"), context)
    execute_assignment(assignment(main, "config['deployment_contract']"), context)
    config = context["config"]
    assert config["vision_max_hz"] == 15
    assert "vision_admission_contract" in config["deployment_contract"]
    contract = config["deployment_contract"]["vision_admission_contract"]
    assert contract["max_hz"] == 15
    assert "admission" in contract["anchor"]
    assert "bootstrap" in contract["scope"]
    assert "publication" in contract
    assert "stop" in contract
    save_run_configuration(tmp_path, config)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["deployment_contract"] == config["deployment_contract"]
    with pytest.raises(RuntimeError, match="changed on resume"):
        save_run_configuration(tmp_path, {**config, "vision_max_hz": 12})


def test_scheduling_helper_is_fingerprinted_from_actual_source(tmp_path):
    main = evaluator_main()
    files = ast.literal_eval(assignment(main, "implementation_files").value)
    name = "coexecution/libero/scheduling.py"
    assert name in files
    source = (ROOT / name).read_bytes()
    copied = tmp_path / name
    copied.parent.mkdir(parents=True)
    copied.write_bytes(source)
    context = {
        "config": {},
        "hashlib": hashlib,
        "root": tmp_path,
        "implementation_files": [name],
    }
    statement = assignment(main, "config['implementation_sha256']")
    execute_assignment(statement, context)
    original = context["config"]["implementation_sha256"][name]
    assert original == hashlib.sha256(source).hexdigest()
    copied.write_bytes(source + b"\n# local contract mutation\n")
    execute_assignment(statement, context)
    assert context["config"]["implementation_sha256"][name] != original


def test_wallclock_import_works_outside_repository_without_runtime_dependencies(
    tmp_path,
):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import inspect
import sys
class RejectRuntime:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'gr00t', 'libero', 'mujoco'}:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, RejectRuntime())
sys.path.insert(0, sys.argv[1])
from libero_wallclock import VisionWorker
assert 'max_hz' in inspect.signature(VisionWorker).parameters
""",
            str(ROOT / "scripts"),
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
