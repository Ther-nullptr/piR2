"""Public command dispatch without importing a GPU or running a model."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from coexecution.profiling import cli

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "command,module,prefix",
    [
        ("worker", "worker", []),
        ("flow", "flow", []),
        ("campaign", "campaign", []),
        ("analyze", "reporting", ["analyze"]),
        ("report", "reporting", ["campaign"]),
    ],
)
def test_subcommands_forward_options_without_loading_other_modules(
    monkeypatch, command, module, prefix
):
    calls = []

    def load(name):
        assert name == f"coexecution.profiling.{module}"
        return SimpleNamespace(main=lambda argv: calls.append(argv))

    monkeypatch.setattr(cli.importlib, "import_module", load)
    cli.main([command, "--help"])
    assert calls == [[*prefix, "--help"]]


def test_top_level_help_and_package_import_need_no_gpu_dependencies():
    code = (
        "import sys; import coexecution.profiling.cli; "
        "assert not ({'torch', 'gr00t', 'matplotlib'} & set(sys.modules))"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
    result = subprocess.run(
        [sys.executable, "scripts/profile_libero_contention.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert all(command in result.stdout for command in cli.COMMANDS)
