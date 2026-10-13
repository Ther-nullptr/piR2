"""Shared labels for contention evidence tables and optional figures."""

COLORS = {"S1": "#D55E00", "S2": "#0072B2", "unassigned": "#999999"}


MODE_NAMES = {
    "vlm_only": "VLM 独跑",
    "vlm_only_cooperative": "VLM 独跑（层边界开销）",
    "dit_only": "DiT 独跑",
    "serial": "串行",
    "concurrent_equal": "同优先级并发",
    "concurrent_dit_priority": "DiT 高优先级",
    "cooperative_dit_priority": "层边界协作让出",
}


LATENCY_FIELDS = [
    ("s1_response_ms", "DiT release→CUDA ready 响应（含 permit 等待）"),
    ("s2_response_ms", "VLM release→CUDA ready 响应（含协作暂停）"),
    ("wall_ms", "整轮 host wall"),
    ("s1_gpu_span_ms", "DiT CUDA 流跨度（不含 permit 等待）"),
    ("s2_gpu_span_ms", "VLM CUDA 流跨度（不含前置等待）"),
]


LATENCY_LABELS = dict(LATENCY_FIELDS)
