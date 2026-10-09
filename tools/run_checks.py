"""Run lightweight contracts only; GPU/model checks require explicit invocation."""

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
        .decode()
        .split("\0")
    )
    for name in tracked:
        if name.endswith(".py"):
            ast.parse((ROOT / name).read_text(), filename=name)
    contracts = [
        "scripts/test_libero_protocol_scheduler.py",
        "scripts/test_libero_wallclock.py",
        "scripts/test_libero_supervisor.py",
        "coexecution/test_timeline.py",
        "coexecution/test_attribution.py",
        "coexecution/test_adapter_lifecycle.py",
        "coexecution/test_groot_optimization.py",
    ]
    tests = [x for x in contracts if (ROOT / x).is_file()]
    if tests:
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *tests], cwd=ROOT, check=True
        )
    if (ROOT / "simulator/model.test.cjs").is_file():
        subprocess.run(
            ["node", "--test", "simulator/model.test.cjs"], cwd=ROOT, check=True
        )
    print("Repository syntax and available scheduling/accounting contracts passed.")


if __name__ == "__main__":
    main()
