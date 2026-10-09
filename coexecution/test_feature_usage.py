"""Lineage accounting checks; no model, GPU, or laboratory data is loaded."""

import copy
import json

import pytest

from coexecution.feature_usage import analyze_episode, analyze_report, main


def feature(sequence, published):
    return {
        "feature_sequence": sequence,
        "t": published,
        "source_tick": sequence,
        "capture_s": published - 0.05,
    }


def request(tick, sequence, use, *, completed=None, start=None):
    return {
        "role": "S1",
        "request_tick": tick,
        "feature_sequence": sequence,
        "feature_source_tick": sequence,
        "feature_capture_s": None,
        "start_s": use - 0.01 if start is None else start,
        "use_s": use,
        "completed_s": use + 0.05 if completed is None else completed,
        "delay_ticks": 1,
    }


def execution(tick, producer, t):
    return {
        "tick": tick,
        "t": t,
        "d": 1,
        "executed_slot": {
            "producer_kind": "request",
            "producer_request_tick": producer,
        },
    }


def episode(*, updates=None, requests=None, ticks=None, valid=True, name="episode-a"):
    return {
        "id": name,
        "duration_s": 1.0,
        "start_s": 10.0,
        "end_s": 11.0,
        "metadata": {"task_id": 42, "control_rate_valid": valid},
        "initial_feature": {
            "feature_sequence": 0,
            "feature_source_tick": 0,
            "feature_capture_s": -0.1,
        },
        "feature_updates": updates or [],
        "requests": requests or [],
        "ticks": ticks or [],
        "delay_transitions": [],
    }


def condition(episodes=None):
    return {
        "id": "parallel/trial",
        "layout": "parallel",
        "condition": "trial",
        "episodes": episodes or [],
    }


def report(*episodes):
    return {"schema_version": 2, "conditions": [condition(list(episodes))]}


def test_empty_report_and_episode_keep_unknown_denominators():
    result = analyze_report({"schema_version": 2, "conditions": []})
    assert result["features"] == result["conditions"] == []
    rows, summary = analyze_episode(condition(), episode())
    assert len(rows) == 1 and rows[0]["bootstrap"]
    assert summary["n_known_inwindow_reads"] == 0
    assert summary["mature_utilization_read"] is None
    assert summary["final_d"] is None


def test_one_last_feature_is_censored_even_when_read():
    rows, summary = analyze_episode(
        condition(), episode(updates=[feature(1, 0.1)], requests=[request(3, 1, 0.2)])
    )
    assert rows[1]["n_reads"] == 1 and rows[1]["right_censored"]
    assert summary["n_mature"] == 0
    assert summary["n_censored_used"] == 1
    assert summary["all_published_utilization_read"] == 1
    assert summary["mature_utilization_read"] is None


def test_actual_source_survives_overwrite_before_plan_start():
    rows, _ = analyze_episode(
        condition(),
        episode(
            updates=[feature(1, 0.1), feature(2, 0.4)],
            requests=[request(3, 1, 0.5, start=0.35)],
        ),
    )
    assert rows[1]["read_request_ticks"] == [3]
    assert rows[1]["n_use_after_overwrite"] == 1
    assert rows[2]["n_reads"] == 0


def test_reuse_counts_requests_separately_from_executed_vectors():
    rows, summary = analyze_episode(
        condition(),
        episode(
            updates=[feature(1, 0.1), feature(2, 0.8)],
            requests=[request(1, 1, 0.2), request(2, 1, 0.4)],
            ticks=[execution(7, 1, 0.3), execution(8, 1, 0.35)],
        ),
    )
    assert (
        rows[1]["n_reads"],
        rows[1]["n_direct_requests"],
        rows[1]["n_execution_ticks"],
    ) == (2, 1, 2)
    assert summary["mature_n_two"] == 1
    assert summary["mature_utilization_direct_observed"] == 1


def test_partial_unknown_sources_are_not_assigned_to_latest_cache():
    incomplete = request(4, 1, 0.5)
    incomplete["completed_s"] = None
    rows, summary = analyze_episode(
        condition(),
        episode(
            updates=[feature(1, 0.1), feature(2, 0.8)],
            requests=[
                request(1, 1, 0.2),
                request(2, None, 0.3),
                request(3, 99, 0.4),
                incomplete,
            ],
            ticks=[execution(4, 2, 0.4), execution(5, 3, 0.5), execution(6, 4, 0.6)],
        ),
    )
    assert summary["n_missing_reads"] == 2
    assert summary["n_unmatched_reads"] == 1
    assert summary["n_known_inwindow_reads"] == 2
    assert summary["n_unmatched_execution_ticks"] == 3
    assert rows[1]["n_reads"] == 1


def test_future_completion_is_counted_but_outside_window_read_is_not():
    rows, summary = analyze_episode(
        condition(),
        episode(
            updates=[feature(1, 0.1), feature(2, 0.98)],
            requests=[
                request(1, 1, 0.95, completed=1.2),
                request(2, 2, 1.0, completed=1.3),
            ],
        ),
    )
    assert rows[1]["n_reads"] == 1
    assert summary["n_read_requests_finishing_after_window"] == 1
    assert summary["n_outside_window_reads"] == 1


def test_bootstrap_reads_and_bootstrap_actions_are_separate():
    ticks = [
        {"tick": 0, "t": 0.01, "d": 1, "executed_slot": {"producer_kind": "bootstrap"}},
        execution(1, 0, 0.15),
        {"tick": 2, "t": 0.2, "d": 1, "fallback": True},
    ]
    rows, summary = analyze_episode(
        condition(), episode(requests=[request(0, 0, 0.05)], ticks=ticks)
    )
    assert rows[0]["n_execution_ticks"] == 1
    assert summary["n_bootstrap_reads"] == 1
    assert summary["n_bootstrap_execution_ticks"] == 1
    assert summary["n_fallback_execution_ticks"] == 1
    assert summary["n_published"] == 0


def test_valid_only_is_filtered_and_condition_counts_are_pooled():
    first = episode(
        updates=[feature(1, 0.1), feature(2, 0.8)],
        requests=[request(0, 1, 0.2)],
        name="first",
    )
    second = episode(
        updates=[feature(1, 0.1), feature(2, 0.3), feature(3, 0.5), feature(4, 0.9)],
        requests=[request(0, 1, 0.2), request(1, 1, 0.25)],
        valid=False,
        name="second",
    )
    summary = analyze_report(report(first, second))["conditions"][0]
    assert summary["n_valid_episodes"] == 1
    assert summary["mature_utilization_read"] == 0.5
    assert summary["mature_mean_reads"] == 0.75
    assert (
        summary["mature_utilization_read"] * summary["mature_mean_reads_if_used"]
        == summary["mature_mean_reads"]
    )
    assert summary["valid_only"]["mature_utilization_read"] == 1


@pytest.mark.parametrize(
    "target", ["features", "requests", "ticks", "episodes", "conditions"]
)
def test_duplicate_identifiers_fail_instead_of_silently_overwriting(target):
    item = episode(
        updates=[feature(1, 0.1)],
        requests=[request(1, 1, 0.2)],
        ticks=[execution(1, 1, 0.3)],
    )
    data = report(item)
    mapping = {
        "features": item["feature_updates"],
        "requests": item["requests"],
        "ticks": item["ticks"],
        "episodes": data["conditions"][0]["episodes"],
        "conditions": data["conditions"],
    }
    mapping[target].append(copy.deepcopy(mapping[target][0]))
    with pytest.raises(ValueError, match="duplicate"):
        analyze_report(data)


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"schema_version": 1, "conditions": []},
        {"schema_version": 2, "conditions": None},
    ],
)
def test_invalid_report_schema_is_explicit(bad):
    with pytest.raises(ValueError):
        analyze_report(bad)


@pytest.mark.parametrize(
    "field,value",
    [
        ("duration_s", float("nan")),
        ("duration_s", -1),
        ("feature_updates", None),
        ("initial_feature", None),
    ],
)
def test_invalid_episode_shape_and_nonfinite_times_fail(field, value):
    item = episode()
    item[field] = value
    with pytest.raises(ValueError):
        analyze_episode(condition(), item)


def test_known_capture_metadata_mismatch_is_an_error():
    item = request(0, 1, 0.2)
    item["feature_capture_s"] = 99
    with pytest.raises(ValueError, match="capture"):
        analyze_episode(
            condition(), episode(updates=[feature(1, 0.1)], requests=[item])
        )


def test_hardware_metadata_does_not_infer_frequency_from_condition_name():
    data = report(episode())
    data["conditions"][0]["hardware"] = {
        "requested_profile": ["arbitrary", 777, 88],
        "role_gpus": {"S1": 4, "S2": 5},
    }
    result = analyze_report(data)["conditions"][0]
    assert result["hardware"]["requested_profile"] == ["arbitrary", 777, 88]
    assert "requested_core_mhz" not in result
    assert analyze_report(report(episode()))["conditions"][0]["hardware"] == {}


def test_cli_uses_explicit_paths_and_writes_only_derived_tables(tmp_path, capsys):
    source = tmp_path / "report.json"
    source.write_text(json.dumps(report(episode())))
    output = tmp_path / "derived"
    main(["--report", str(source), "--output", str(output)])
    assert sorted(p.name for p in output.iterdir()) == [
        "feature-metrics.json",
        "feature-summary.csv",
        "per-feature.csv",
    ]
    derived = json.loads((output / "feature-metrics.json").read_text())
    assert derived["source_report"] == source.name
    assert derived["validation"]["raw_logs_verified"] == 0
    assert json.loads(capsys.readouterr().out)["episodes"] == 1


def test_optional_raw_audit_checks_source_ids_and_detects_mismatch(tmp_path):
    item = episode(requests=[request(0, 0, 0.05)])
    raw_dir = tmp_path / "parallel" / "trial"
    raw_dir.mkdir(parents=True)
    path = raw_dir / "episode-a-trace.jsonl"
    raw = [
        {
            "kind": "action_completed",
            "request_tick": 0,
            "feature_sequence": 0,
            "feature_source_tick": 0,
            "feature_capture_s": None,
        },
        {
            "kind": "request_completed",
            "audit": {"request_tick": 0},
            "client_timing": {"feature_sequence": 0, "plan_rpc_started_s": 10.05},
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in raw))
    result = analyze_report(report(item), raw_root=tmp_path)
    assert result["validation"]["raw_logs_verified"] == 1
    raw[0]["feature_sequence"] = 99
    path.write_text("\n".join(json.dumps(r) for r in raw))
    with pytest.raises(ValueError, match="raw"):
        analyze_report(report(item), raw_root=tmp_path)


def test_unknown_validity_is_not_treated_as_a_valid_episode():
    item = episode(valid=None)
    summary = analyze_report(report(item))["conditions"][0]
    assert summary["n_valid_episodes"] == 0
    assert summary["n_unknown_validity_episodes"] == 1


def test_missing_provenance_marks_read_accounting_incomplete():
    _, summary = analyze_episode(condition(), episode(requests=[request(0, None, 0.1)]))
    assert not summary["read_accounting_complete"]


def test_optional_raw_root_rejects_traversal(tmp_path):
    data = report(episode())
    data["conditions"][0]["layout"] = ".."
    with pytest.raises(ValueError, match="path names"):
        analyze_report(data, raw_root=tmp_path)


def test_duplicate_bootstrap_and_published_feature_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        analyze_episode(condition(), episode(updates=[feature(0, 0.1)]))


def test_nonfinite_request_timestamp_is_rejected():
    item = request(0, 1, float("nan"))
    with pytest.raises(ValueError, match="finite"):
        analyze_episode(condition(), episode(requests=[item]))
