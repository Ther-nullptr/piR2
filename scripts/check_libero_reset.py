"""Trace reset state before/after cleaning accumulated property samplers."""

import hashlib
import json
from pathlib import Path

from check_libero_timing_gpu import reset
from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv
from libero.libero import benchmark, get_libero_path

root = Path(__file__).resolve().parents[1]
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(4)
state = suite.get_task_init_states(4)[0]
env = LiberoEnv(
    str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
    task.language,
)
rows = []


def read(mode):
    obs = reset(env, state, 5000)
    domain = env._env.env
    sim = domain.sim
    record = {
        "mode": mode,
        "property_samplers": len(domain.object_property_initializers),
        "fixture_hash": hashlib.sha256(
            sim.model.body_pos.tobytes() + sim.model.body_quat.tobytes()
        ).hexdigest(),
        "state_hash": hashlib.sha256(sim.get_state().flatten().tobytes()).hexdigest(),
        "image_hash": hashlib.sha256(obs["video.image"].tobytes()).hexdigest(),
    }
    rows.append(record)
    print(record, flush=True)


try:
    for _ in range(3):
        read("original")
    domain = env._env.env
    original = domain._load_model

    def clean_reload():
        domain.object_property_initializers.clear()
        return original()

    domain._load_model = clean_reload
    for _ in range(3):
        read("clear_on_model_reload")
finally:
    env.close()
(root / "artifacts/libero-protocols/reset-repeatability.json").write_text(
    json.dumps(rows, indent=2) + "\n"
)
assert (
    len({(x["fixture_hash"], x["state_hash"], x["image_hash"]) for x in rows[3:]}) == 1
)
print("RESET_REPEATABILITY_FIXED", flush=True)
