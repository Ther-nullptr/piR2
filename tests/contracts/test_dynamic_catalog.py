"""Benchmark coverage and provenance without policy or simulator runtimes."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from coexecution.libero_dynamic.catalog import (
    build_inventory,
    expand_conditions,
    read_reviewed_config,
)

ROOT = Path(__file__).resolve().parents[2]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
LEVELS = ("l1", "l2", "l3")


@pytest.fixture
def configs():
    configs = {}
    for level in LEVELS:
        tasks = {}
        for suite in SUITES:
            tasks[suite] = {}
            for task_id in range(10):
                variants = []
                for index in range(5):
                    variants.append(
                        {
                            "variant_id": f"{suite}_task{task_id:02d}__{level}__traj{index:02d}",
                            "variant_name": f"traj{index:02d}",
                            "motion": {
                                "type": "zigzag_waypoints",
                                "editor_motion_type": "conveyor_pingpong",
                                "alpha_mps": 0.06,
                            },
                            "target_objects": [
                                {
                                    "target_object_name": "bowl_1",
                                    "target_joint_name": "bowl_1_joint0",
                                    "motion": {"type": "zigzag_waypoints"},
                                },
                                {
                                    "target_object_name": "plate_1",
                                    "motion": {"type": "polynomial_spline"},
                                },
                            ],
                        }
                    )
                tasks[suite][task_id] = {
                    "enabled": True,
                    "target_object_name": "bowl_1",
                    "target_joint_name": "bowl_1_joint0",
                    "target_kind": "movable_object",
                    "activation_variants": variants,
                    "initial_scene_adjustments": {"settle_steps": 500},
                }
        configs[level] = {"enabled": True, "tasks": tasks}
    return configs


def test_full_population_has_exactly_1800_conditions_and_stable_identity(configs):
    rows = expand_conditions(configs)
    assert len(rows) == 1800
    assert len({row["condition_id"] for row in rows}) == 1800
    assert len({row["item_id"] for row in rows}) == 600
    selected = [r for r in rows if r["item_id"] == "libero_spatial_task00__l1__traj00"]
    assert [r["condition_id"] for r in selected] == [
        "libero_spatial_task00__l1__traj00__s003",
        "libero_spatial_task00__l1__traj00__s006",
        "libero_spatial_task00__l1__traj00__s009",
    ]
    assert [r["speed_mps"] for r in selected] == [0.03, 0.06, 0.09]
    assert all(r["seed"] == 0 and r["init_state_id"] == 0 for r in rows)
    assert all(r["pre_policy_dynamic_steps"] == 40 for r in rows)
    reordered = {k: configs[k] for k in reversed(LEVELS)}
    assert expand_conditions(reordered) == rows


def test_selection_is_explicit_and_all_sources_still_validated(configs):
    rows = expand_conditions(configs, suites=["libero_spatial"])
    assert len(rows) == 450
    assert {r["suite"] for r in rows} == {"libero_spatial"}
    del configs["l3"]["tasks"]["libero_goal"][9]
    with pytest.raises(ValueError, match="task coverage"):
        expand_conditions(configs, suites=["libero_spatial"])


@pytest.mark.parametrize(
    "defect",
    ["level", "suite", "task", "variant", "duplicate", "identity", "disabled", "order"],
)
def test_missing_or_duplicate_conditions_never_silently_skip(configs, defect):
    task = configs["l1"]["tasks"]["libero_spatial"][0]
    if defect == "level":
        del configs["l3"]
    elif defect == "suite":
        del configs["l1"]["tasks"]["libero_object"]
    elif defect == "task":
        del configs["l1"]["tasks"]["libero_object"][4]
    elif defect == "variant":
        task["activation_variants"].pop()
    elif defect == "duplicate":
        task["activation_variants"][4] = copy.deepcopy(task["activation_variants"][0])
    elif defect == "identity":
        task["activation_variants"][0]["variant_id"] = "another_task"
    elif defect == "disabled":
        task["enabled"] = False
    elif defect == "order":
        task["activation_variants"].reverse()
    with pytest.raises(ValueError):
        expand_conditions(configs)


@pytest.mark.parametrize(
    "speeds",
    [[0], [-0.03], [float("nan")], [float("inf")], [True], [0.03, 0.03], [], [0.035]],
)
def test_speed_grid_rejects_static_alias_invalid_and_nonbenchmark_values(
    configs, speeds
):
    with pytest.raises(ValueError, match="speed"):
        expand_conditions(configs, speeds=speeds)


def test_fixture_identity_and_independent_target_motion_survive(configs):
    task = configs["l2"]["tasks"]["libero_goal"][0]
    task["target_kind"] = "fixture"
    task["target_body_name"] = "cabinet_main"
    variant = task["activation_variants"][0]
    variant["target_object_name"] = "cabinet_1"
    variant["target_joint_name"] = "cabinet_hinge"
    variant["motion"]["type"] = "body_yaw_polynomial_spline"
    variant["target_objects"][0]["target_object_name"] = "cabinet_1"
    variant["target_objects"][0]["motion"] = copy.deepcopy(variant["motion"])
    variant["initial_scene_adjustments"] = {"settle_steps": 80}
    rows = expand_conditions(configs)
    row = next(r for r in rows if r["item_id"] == "libero_goal_task00__l2__traj00")
    assert row["target_object_name"] == "cabinet_1"
    assert row["target_joint_name"] == "cabinet_hinge"
    assert row["target_body_name"] == "cabinet_main"
    assert row["target_kind"] == "fixture"
    assert row["motion"]["type"] == "body_yaw_polynomial_spline"
    assert row["motion"]["editor_motion_type"] == "conveyor_pingpong"
    assert row["target_objects"] == variant["target_objects"]
    assert row["initial_scene_adjustments"] == {"settle_steps": 80}
    row["target_objects"][0]["motion"]["type"] = "changed"
    assert (
        variant["target_objects"][0]["motion"]["type"] == "body_yaw_polynomial_spline"
    )


def test_effective_motion_follows_first_target_without_losing_raw_variant(configs):
    variant = configs["l2"]["tasks"]["libero_spatial"][7]["activation_variants"][2]
    variant["motion"] = {"type": "zigzag_waypoints", "waypoints": [[0, 0], [1, 0]]}
    effective = {"type": "polynomial_spline", "waypoints": [[0, 0], [0, 1]]}
    variant["target_objects"][0]["motion"] = effective
    row = next(
        r
        for r in expand_conditions(configs)
        if r["item_id"] == "libero_spatial_task07__l2__traj02"
    )
    assert row["motion"] == effective
    assert row["raw_variant_motion"] == variant["motion"]


def test_verified_loader_hashes_raw_bytes_before_parsing(tmp_path):
    path = tmp_path / "config.yaml"
    raw = b"enabled: true\ntasks: {}\n"
    path.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    assert read_reviewed_config(path, expected) == {"enabled": True, "tasks": {}}
    path.write_bytes(raw + b"# modified\n")
    with pytest.raises(ValueError, match="SHA256"):
        read_reviewed_config(path, expected)
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        read_reviewed_config(path, expected)


def test_upstream_cannot_self_assert_a_new_hash(tmp_path):
    manifest = json.loads((ROOT / "configs/libero/dynamic-benchmark.json").read_text())
    for source in manifest["configs"].values():
        path = tmp_path / source["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("enabled: true\n")
        source["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "dynamic-benchmark.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA256"):
        build_inventory(tmp_path)


def test_import_and_help_do_not_load_policy_or_simulator_dependencies():
    program = """
import sys
class RejectRuntime:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'libero', 'mujoco', 'yaml'}:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, RejectRuntime())
from coexecution.libero_dynamic.__main__ import main
main(['inventory', '--help'])
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--upstream" in result.stdout


def test_single_fixture_without_target_objects_is_preserved(configs):
    task = configs["l1"]["tasks"]["libero_goal"][0]
    task["target_object_name"] = "wooden_cabinet_1"
    task["target_joint_name"] = "wooden_cabinet_1_middle_level"
    task["target_body_name"] = "wooden_cabinet_1_main"
    task["target_kind"] = "articulated_fixture_body"
    for variant in task["activation_variants"]:
        del variant["target_objects"]
        variant["motion"] = {
            "type": "body_yaw_pingpong",
            "alpha_yaw_rad_per_s": 0.2,
            "alpha_mps": 0.055,
        }
    row = next(
        r
        for r in expand_conditions(configs)
        if r["item_id"] == "libero_goal_task00__l1__traj00"
    )
    assert row["target_body_name"] == "wooden_cabinet_1_main"
    assert row["motion"]["alpha_yaw_rad_per_s"] == 0.2
    assert "target_objects" not in row
    del task["target_object_name"]
    with pytest.raises(ValueError, match="target identity"):
        expand_conditions(configs)
