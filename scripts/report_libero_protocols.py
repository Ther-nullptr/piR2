"""Aggregate success and measured timing without averaging episode percentiles."""

import argparse
import json
from pathlib import Path

from libero_protocol_scheduler import summarize_values


def report(root):
    rows = []
    for protocol in ["algorithm", "deployment"]:
        for variant in ["pir2", "flow"]:
            folder = root / f"{protocol}-{variant}"
            summary = json.loads((folder / "summary.json").read_text())
            assert summary["complete"]
            episodes = [
                json.loads(x)
                for x in (folder / "episodes.jsonl").read_text().splitlines()
            ]
            action_ms = []
            vlm_ms = []
            age_ms = []
            jitter_ms = []
            for path in folder.glob("*-trace.jsonl"):
                for line in path.open():
                    event = json.loads(line)
                    kind = event["kind"]
                    if kind in ["request_completed", "unused_result_at_episode_end"]:
                        timing = event["client_timing"]
                        action_ms.append(
                            1000 * (timing["completed_s"] - timing["submitted_s"])
                        )
                        if "cache_age_at_request_ms" in timing:
                            age_ms.append(timing["cache_age_at_request_ms"])
                    elif kind == "vision_request":
                        vlm_ms.append(event["rpc_seconds"] * 1000)
                    elif kind == "control_tick":
                        jitter_ms.append(event["lateness_s"] * 1000)
            row = {
                "protocol": protocol,
                "variant": variant,
                "successes": summary["successes"],
                "episodes": summary["episodes"],
                "success_rate": summary["success_rate"],
                "action_rpc_ms": summarize_values(action_ms),
                "vlm_rpc_ms": summarize_values(vlm_ms),
                "cache_age_at_action_request_ms": summarize_values(age_ms),
                "control_lateness_ms": summarize_values(jitter_ms),
            }
            if protocol == "deployment":
                row.update(
                    actual_control_hz=summarize_values(
                        [
                            x["actual_control_hz"]
                            for x in episodes
                            if x["actual_control_hz"] is not None
                        ]
                    ),
                    action_request_hz=summarize_values(
                        [
                            x.get(
                                "action_request_hz",
                                x["action_requests"] / x["controlled_wall_seconds"],
                            )
                            for x in episodes
                        ]
                    ),
                    strict_timing_valid_episodes=sum(
                        x["control_rate_valid"] for x in episodes
                    ),
                    mean_rate_valid_episodes=sum(
                        x["mean_control_rate_valid"] for x in episodes
                    ),
                    control_deadline_misses=sum(
                        x["control_deadline_misses_over5ms"] for x in episodes
                    ),
                    action_request_deadline_misses=sum(
                        x["action_request_deadline_misses"] for x in episodes
                    ),
                    fallback_ticks=sum(x["fallback_ticks"] for x in episodes),
                    expired_action_slots=sum(
                        x["expired_action_slots"] for x in episodes
                    ),
                )
            rows.append(row)
    (root / "report.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# LIBERO 两类时钟对照结果",
        "",
        "算法对照：固定视觉延迟150ms、动作延迟50ms；状态保持当前。部署对照：目标20Hz控制，真实异步VLM与动作推理。所有成功率保留全部回合，超时和时序不合格回合不被静默剔除。",
        "",
        "| 协议 | 模型 | 成功率 | 控制Hz均值 | 动作RPC p95 ms | VLM RPC p95 ms | 缓存年龄 p95 ms | 严格时序合格 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    def val(row, key, stat):
        return f"{row[key][stat]:.2f}" if row.get(key) else "—"

    for row in rows:
        valid = (
            f"{row['strict_timing_valid_episodes']}/{row['episodes']}"
            if "strict_timing_valid_episodes" in row
            else "不使用墙钟"
        )
        lines.append(
            f"| {row['protocol']} | {row['variant']} | {row['successes']}/{row['episodes']} ({100 * row['success_rate']:.1f}%) | {val(row, 'actual_control_hz', 'mean')} | {val(row, 'action_rpc_ms', 'p95')} | {val(row, 'vlm_rpc_ms', 'p95')} | {val(row, 'cache_age_at_action_request_ms', 'p95')} | {valid} |"
        )
    lines += [
        "",
        "延迟分位数从原始逐请求记录汇总，不对各回合p95做平均。控制Hz与策略请求Hz分别记录；两模型使用相同GPU布局。严格时序合格要求平均频率误差≤2%，控制迟到和相邻周期偏差均≤5ms。初始化、GPU预热和校准在计时外单独记录。",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    report(parser.parse_args().root)
