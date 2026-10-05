"""Fetch inference weights and the full official Spatial training dataset."""

import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "artifacts/libero-pir2").mkdir(parents=True, exist_ok=True)
api = HfApi()
model_repo = "nvidia/GR00T-N1.7-LIBERO"
model_revision = "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21"
patterns = [
    "libero_spatial/config.json",
    "libero_spatial/embodiment_id.json",
    "libero_spatial/model-*.safetensors",
    "libero_spatial/model.safetensors.index.json",
    "libero_spatial/processor_config.json",
    "libero_spatial/statistics.json",
    "libero_spatial/experiment_cfg/*",
]
print("Downloading official Spatial inference checkpoint", flush=True)
snapshot_download(
    model_repo,
    revision=model_revision,
    allow_patterns=patterns,
    local_dir=ROOT / "models/GR00T-N1.7-LIBERO",
    max_workers=4,
)
print("MODEL_DOWNLOAD_COMPLETE", flush=True)
repo = "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot"
dataset_revision = "bf14d6258218d12c2e3c1a3b9922e163cdf6455d"
info = api.dataset_info(repo, revision=dataset_revision, files_metadata=True)
manifest = {
    "model_repo": model_repo,
    "model_revision": model_revision,
    "dataset_repo": repo,
    "dataset_revision": info.sha,
    "dataset_files": [{"path": f.rfilename, "bytes": f.size} for f in info.siblings],
}
(ROOT / "artifacts/libero-pir2/assets-manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
print("Downloading all Spatial demonstrations", info.sha, flush=True)
snapshot_download(
    repo,
    repo_type="dataset",
    revision=info.sha,
    local_dir=ROOT / "datasets/groot-libero-spatial",
    max_workers=4,
)
print("DATASET_DOWNLOAD_COMPLETE", flush=True)
shutil.copyfile(
    ROOT / "upstream/learning/Isaac-GR00T/examples/LIBERO/modality.json",
    ROOT / "datasets/groot-libero-spatial/meta/modality.json",
)
