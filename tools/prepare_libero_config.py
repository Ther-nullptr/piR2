"""Write an isolated LIBERO config from installed package metadata and local assets."""

import argparse
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    assets = args.assets.expanduser().resolve()
    if not assets.is_dir():
        raise FileNotFoundError(f"Download LIBERO assets first: {assets}")
    package = importlib.metadata.distribution("hf-libero")
    benchmark = Path(package.locate_file("libero/libero")).resolve()
    for directory in ["bddl_files", "init_files"]:
        if not (benchmark / directory).is_dir():
            raise FileNotFoundError(benchmark / directory)
    config = {
        "assets": str(assets),
        "benchmark_root": str(benchmark),
        "bddl_files": str(benchmark / "bddl_files"),
        "init_states": str(benchmark / "init_files"),
        "datasets": str(ROOT / "datasets/libero"),
    }
    target = ROOT / ".libero-config-pi05/config.yaml"
    if target.exists():
        raise FileExistsError(f"Existing config left untouched: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # JSON is a YAML subset; no simulator import or GPU/model execution needed.
    target.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Created {target}; set LIBERO_CONFIG_PATH={target.parent}")


if __name__ == "__main__":
    main()
