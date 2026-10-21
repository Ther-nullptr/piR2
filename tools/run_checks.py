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
        "tests/contracts/test_adapter_lifecycle.py",
        "tests/contracts/test_groot_optimization.py",
        "tests/contracts/test_input_preparation.py",
        "tests/contracts/test_tokenization.py",
        "tests/contracts/test_image_preparation.py",
        "tests/contracts/test_compiled_host_guards.py",
        "tests/contracts/test_vlm_patching.py",
        "tests/contracts/test_policy_vlm.py",
        "tests/contracts/test_policy_measurement.py",
        "tests/contracts/test_prepared_inference.py",
        "tests/contracts/test_libero_profiling.py",
        "tests/contracts/test_replay_evidence.py",
        "tests/contracts/test_replay_live.py",
        "tests/contracts/test_replay_producers.py",
        "tests/contracts/test_replay_stream.py",
        "tests/contracts/test_runtime_receipt.py",
        "tests/contracts/test_libero_experiments.py",
        "tests/contracts/test_dynamic_catalog.py",
        "tests/contracts/test_dynamic_policy.py",
        "tests/contracts/test_responsive_scheduling.py",
        "tests/contracts/test_synchronous_execution.py",
        "tests/contracts/test_one_step_async.py",
        "tests/contracts/test_chunked_async.py",
        "tests/contracts/test_async_replay_evidence.py",
        "tests/contracts/test_feature_transport.py",
        "tests/contracts/test_resident_features.py",
        "tests/contracts/test_cuda_submission.py",
        "tests/contracts/test_vision_admission_configuration.py",
        "tests/contracts/test_dit_graph_buckets.py",
        "tests/contracts/test_action_graph.py",
        "tests/contracts/test_action_kv_reuse.py",
        "tests/contracts/test_optimization_imports.py",
        "tests/contracts/test_backend_capabilities.py",
        "tests/contracts/test_streaming_components.py",
        "tests/contracts/test_tactic_profiles.py",
        "tests/contracts/test_operator_validation.py",
        "coexecution/test_queue_report.py",
        "coexecution/test_feature_usage.py",
        "tests/contracts/test_libero_inference_backend.py",
        "tests/contracts/test_libero_model_loading.py",
        "tests/contracts/test_libero_imports.py",
        "tests/contracts/test_contention_cli.py",
        "tests/contracts/test_flow_contention.py",
        "tests/contracts/test_contention_worker.py",
        "tests/contracts/test_contention_campaign.py",
        "tests/contracts/test_contention_analysis.py",
        "simulator/test_serve.py",
    ]
    tests = [x for x in contracts if (ROOT / x).is_file()]
    if tests:
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *tests], cwd=ROOT, check=True
        )
    if (ROOT / "simulator/model.test.cjs").is_file():
        subprocess.run(
            [
                "node",
                "--test",
                *[
                    str(p.relative_to(ROOT))
                    for p in sorted((ROOT / "simulator").glob("*.test.cjs"))
                ],
            ],
            cwd=ROOT,
            check=True,
        )
    print("Repository syntax and available scheduling/accounting contracts passed.")


if __name__ == "__main__":
    main()
