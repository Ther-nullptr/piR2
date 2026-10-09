"""Measured accounting checks; no policy/model inference is performed."""

import pytest

from coexecution.queue_report import integrate_telemetry, summarize_episode


def event(kind, t_s, **fields):
    return {"kind": kind, "t_s": t_s, **fields}


def test_requests_crossing_control_end_are_censored_not_complete_samples():
    rows = [
        event("control_start", 10, period_s=0.05),
        event("action_submitted", 10.1, request_tick=0),
        event(
            "action_completed",
            10.3,
            request_tick=0,
            request_elapsed_s=0.2,
            valid_start_tick=2,
            valid_end_tick=4,
        ),
        event("action_submitted", 10.8, request_tick=4),
        event(
            "action_completed",
            11.2,
            request_tick=4,
            request_elapsed_s=0.4,
            valid_start_tick=6,
            valid_end_tick=8,
        ),
        event("vision_request", 9.9, started_s=9.5, completed_s=9.9, rpc_seconds=0.4),
        event(
            "vision_request", 10.4, started_s=10.1, completed_s=10.4, rpc_seconds=0.3
        ),
        event(
            "vision_request", 11.1, started_s=10.9, completed_s=11.1, rpc_seconds=0.2
        ),
        event("control_end", 11, control_steps=6),
    ]
    out = summarize_episode(rows, {})
    assert out["metrics"]["action_rpc_ms"]["count"] == 1
    assert out["metrics"]["action_rpc_ms"]["mean"] == pytest.approx(200)
    assert out["metrics"]["vision_rpc_ms"]["count"] == 1
    assert out["metrics"]["action_censored_requests"] == 1
    assert out["metrics"]["vision_censored_requests"] == 1
    assert max(r["end_s"] for r in out["requests"]) == 1


def test_unneeded_future_slots_are_censored_and_never_expired():
    rows = [
        event("control_start", 0, period_s=0.05),
        event("action_submitted", 0.01, request_tick=0),
        event(
            "action_completed",
            0.21,
            request_tick=0,
            request_elapsed_s=0.2,
            valid_start_tick=2,
            valid_end_tick=7,
        ),
        event("control_end", 0.25, control_steps=5),
        event(
            "unused_result_at_episode_end",
            0.3,
            expired_before_episode_end=3,
            unneeded_future_slots=2,
            audit={"request_tick": 0},
        ),
    ]
    out = summarize_episode(rows, {})["metrics"]
    assert out["expired_slots"] == 3
    assert out["censored_future_slots"] == 2
    assert out["due_generated_slots"] == 3
    assert out["expired_slot_fraction"] == 1


def test_energy_interpolates_window_boundaries_and_ignores_gaps_between_episodes():
    samples = [
        {
            "t_s": t,
            "gpu": 2,
            "power_w": 10 + 10 * t,
            "sm_mhz": 1000,
            "energy_mj": (10 * t + 5 * t * t) * 1000,
        }
        for t in range(5)
    ]
    # 0.5..1.5: 20 J; 2.5..3.5: 40 J. Never integrate the reset gap.
    out = integrate_telemetry(samples, [(0.5, 1.5), (2.5, 3.5)], [2])
    assert out["sampled_energy_j"] == pytest.approx(60)
    assert out["counter_energy_j"] == pytest.approx(60)
    assert out["coverage_fraction"] == 1
    assert out["average_power_w"] == pytest.approx(30)


def test_missing_telemetry_is_not_extrapolated_or_reported_as_full_energy():
    samples = [
        {"t_s": 1, "gpu": 2, "power_w": 100},
        {"t_s": 2, "gpu": 2, "power_w": 100},
    ]
    out = integrate_telemetry(samples, [(0, 3)], [2])
    assert out["sampled_energy_j"] == pytest.approx(100)
    assert out["coverage_fraction"] == pytest.approx(1 / 3)
    assert out["counter_energy_j"] is None
    assert out["average_power_w"] is None


def test_due_deadline_miss_remains_observed_when_result_finishes_after_end():
    rows = [
        event("control_start", 10, period_s=0.05),
        event("action_submitted", 10.1, request_tick=0, deadline_s=10.8),
        event("action_completed", 11.2, request_tick=0, request_elapsed_s=1.1),
        event("control_end", 11, control_steps=20),
    ]
    out = summarize_episode(rows, {})
    assert out["metrics"]["deadline_misses"] == 1
    assert out["miss_events"][0]["t"] == pytest.approx(0.8)


def test_consumed_feature_comes_from_completion_and_client_timing_not_submission():
    rows = [
        event(
            "control_start",
            10,
            period_s=0.05,
            initial_capture_s=9.8,
            feature_sequence=0,
            feature_capture_s=9.8,
        ),
        event(
            "action_submitted",
            10.01,
            request_tick=0,
            feature_sequence=90,
            feature_capture_s=9.99,
            state_capture_s=9.9,
            delay_ticks=1,
        ),
        event(
            "action_completed",
            10.04,
            request_tick=0,
            feature_sequence=3,
            feature_source_tick=2,
            feature_capture_s=9.7,
            state_capture_s=9.9,
        ),
        event(
            "request_completed",
            10.05,
            audit={"request_tick": 0},
            client_timing={
                "plan_rpc_started_s": 10.02,
                "feature_sequence": 3,
                "feature_capture_s": 9.7,
                "state_capture_s": 9.9,
            },
        ),
        event("action_submitted", 10.06, request_tick=1, feature_sequence=99),
        event("control_end", 10.2, control_steps=4),
    ]
    out = summarize_episode(rows, {})
    request, unknown = [r for r in out["requests"] if r["role"] == "S1"]
    assert request["feature_sequence"] == 3
    assert request["feature_capture_s"] == pytest.approx(-0.3)
    assert request["state_capture_s"] == pytest.approx(-0.1)
    assert request["use_s"] == pytest.approx(0.02)
    assert unknown["feature_sequence"] is None
    assert unknown["feature_capture_s"] is None
    assert out["initial_capture_s"] == pytest.approx(-0.2)


def test_feature_publications_match_rpc_identity_and_delay_changes_at_consumption():
    rows = [
        event("control_start", 10, period_s=0.05, initial_delay_ticks=1),
        event("action_submitted", 10.001, request_tick=0, deadline_s=10.05),
        event("action_completed", 10.0504, request_tick=0),
        event(
            "request_completed",
            10.055,
            deadline_miss=True,
            audit={"request_tick": 0},
            client_timing={
                "submitted_s": 10.001,
                "completed_s": 10.0504,
                "deadline_s": 10.05,
            },
        ),
        event(
            "feature_published",
            10.062,
            source_tick=0,
            capture_s=9.8,
            rpc_completed_s=10.06,
            feature_sequence=7,
        ),
        event(
            "vision_request",
            10.06,
            source_tick=0,
            capture_s=9.8,
            started_s=10.002,
            completed_s=10.06,
            rpc_seconds=0.058,
        ),
        event("control_end", 10.2, control_steps=4),
    ]
    out = summarize_episode(rows, {})
    s2 = next(r for r in out["requests"] if r["role"] == "S2")
    assert s2["feature_sequence"] == 7
    assert s2["published_s"] == pytest.approx(0.062)
    assert out["feature_updates"][0]["capture_s"] == pytest.approx(-0.2)
    transition = out["delay_transitions"][0]
    assert transition["t"] == pytest.approx(0.055)
    assert (transition["before"], transition["after"]) == (1, 2)
    assert transition["overshoot_ms"] == pytest.approx(0.4)
    assert out["miss_events"][0]["overshoot_ms"] == pytest.approx(0.4)


def test_no_figures_exports_dynamic_report_without_matplotlib(tmp_path, monkeypatch):
    import builtins
    import json

    from coexecution.queue_report import export_report

    source = tmp_path / "run"
    source.mkdir()
    (source / "hardware.json").write_text(
        json.dumps(
            {
                "layout": "single",
                "condition": "example",
                "inference_gpus": [3],
                "role_gpus": {"S1": 3, "S2": 3},
                "returncode": 0,
            }
        )
    )
    rows = [
        event("control_start", 10, period_s=0.05),
        event("control_end", 10.1, control_steps=2),
    ]
    (source / "task0-episode0-trace.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith("matplotlib"), (
            "no-figures must not import matplotlib"
        )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    report = export_report(source, tmp_path / "report", no_figures=True)
    html = (tmp_path / "report" / "index.html").read_text()
    assert report["schema_version"] == 2
    assert "MeasuredReplay" in html
    assert "/*MEASURED_MODEL*/" not in html
    assert "pir2:load-solo-calibration" in html
    assert not list((tmp_path / "report").glob("*.png"))
    assert not list((tmp_path / "report").glob("*.svg"))


def test_tick_aoi_tracks_executed_producer_not_latest_cache_and_fallback_is_null():
    rows = [
        event(
            "control_start",
            10,
            period_s=0.05,
            initial_capture_s=9.8,
            feature_sequence=0,
            feature_capture_s=9.8,
        ),
        event("action_submitted", 10.01, request_tick=0),
        event(
            "action_completed",
            10.04,
            request_tick=0,
            feature_sequence=0,
            feature_capture_s=9.8,
            state_capture_s=9.9,
        ),
        event("feature_published", 10.05, feature_sequence=1, capture_s=10.02),
        event(
            "control_tick",
            10.1,
            tick=2,
            executed_slot={"producer_kind": "request", "producer_request_tick": 0},
        ),
        event(
            "control_tick",
            10.15,
            tick=3,
            fallback=True,
            executed_slot={"producer_kind": "fallback"},
        ),
        event("control_end", 10.2, control_steps=4),
    ]
    out = summarize_episode(rows, {})
    tick = out["ticks"][0]
    assert tick["latest_cache_age_ms"] == pytest.approx(80)
    assert tick["executed_image_age_ms"] == pytest.approx(300)
    assert tick["executed_state_age_ms"] == pytest.approx(200)
    assert out["ticks"][1]["executed_image_age_ms"] is None
    assert out["metrics"]["executed_image_age_ms"]["count"] == 1
