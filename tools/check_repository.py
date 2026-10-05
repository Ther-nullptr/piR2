"""Reject generated assets, secrets, local paths and dangling documentation links."""

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "artifacts",
    "models",
    "datasets",
    "outputs",
    "papers",
    "upstream",
    "third_party",
    ".local",
    "node_modules",
    "__pycache__",
    "runs",
    "logs",
    "profiles",
}
DOCUMENTS = {
    "environment.md",
    "upstream.md",
    "libero.md",
    "timing-protocols.md",
    "coexecution.md",
}


def main():
    names = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
        .decode()
        .split("\0")
    )
    errors = []
    for name in filter(None, names):
        path = ROOT / name
        if FORBIDDEN.intersection(path.relative_to(ROOT).parts):
            errors.append(f"Generated/dependency path: {name}")
        if name.startswith("docs/") and name.removeprefix("docs/") not in DOCUMENTS:
            errors.append(f"Document outside allowlist: {name}")
        if path.is_symlink() or path.stat().st_size > 1_000_000:
            errors.append(f"Symlink or oversized artifact: {name}")
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            errors.append(f"Unexpected binary: {name}")
            continue
        if re.search(r"(?:hf_|ghp_|github_pat_)[A-Za-z0-9_]{20,}", text):
            errors.append(f"Possible credential: {name}")
        if re.search(r"/home/[A-Za-z0-9_-]+/", text):
            errors.append(f"Machine-specific home path: {name}")
        if path.suffix == ".md":
            prose = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
            for link in re.findall(r"\]\(([^)]+)\)", prose):
                target = link.split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                resolved = (path.parent / unquote(target)).resolve()
                if not resolved.is_relative_to(ROOT) or not resolved.exists():
                    errors.append(
                        f"Missing/nonlocal documentation link: {name}: {target}"
                    )
    if errors:
        raise SystemExit("\n".join(errors))
    print("Tracked-file hygiene, secret patterns and local document links passed.")


if __name__ == "__main__":
    main()
