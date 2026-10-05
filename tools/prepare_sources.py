"""Fetch pinned source repositories; never reset an existing working tree."""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(path, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(path), *args], check=check, capture_output=True, text=True
    )


def prepare(spec):
    target = ROOT / spec["path"]
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", spec["url"], str(target)], check=True
        )
        git(target, "checkout", "--detach", spec["revision"])
    if git(target, "rev-parse", "HEAD").stdout.strip() != spec["revision"]:
        raise RuntimeError(f"Unexpected source revision in {target}; left untouched")
    if "submodule" in spec:
        module = target / spec["submodule"]
        if not (module / ".git").exists():
            git(target, "submodule", "update", "--init", "--", spec["submodule"])
        target = target / spec["submodule"]
        if (
            git(target, "rev-parse", "HEAD").stdout.strip()
            != spec["submodule_revision"]
        ):
            raise RuntimeError(f"Unexpected submodule revision in {target}")
    for patch in spec["patches"]:
        path = ROOT / patch
        if (
            git(
                target, "apply", "--reverse", "--check", str(path), check=False
            ).returncode
            == 0
        ):
            print(f"Already applied: {patch}")
        else:
            git(target, "apply", "--check", str(path))
            git(target, "apply", str(path))
            print(f"Applied: {patch}")
    print(f"Verified {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    specs = json.loads((ROOT / "sources.lock.json").read_text())["sources"]
    parser.add_argument("--component", choices=sorted(specs), required=True)
    args = parser.parse_args()
    prepare(specs[args.component])


if __name__ == "__main__":
    main()
