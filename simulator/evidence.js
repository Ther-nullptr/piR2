// Curated historical SO100 replay snapshot, not LIBERO results or hardware predictions.
window.VLAEvidence = {
  fusion: [
    {
      label: "BF16 baseline",
      s2: 23.8626975,
      pair: 39.546755,
      exact: false,
    },
    {
      label: "视觉 RoPE 融合",
      s2: 21.631955,
      pair: 37.9440345,
      exact: true,
    },
    {
      label: "视觉 + 文本 RoPE",
      s2: 21.227094,
      pair: 37.5253685,
      exact: true,
    },
    {
      label: "RMSNorm 部分融合",
      s2: 23.729938500000003,
      pair: 39.5103725,
      exact: true,
    },
    {
      label: "RoPE + RMSNorm",
      s2: 20.742588499999997,
      pair: 38.037889,
      exact: true,
    },
  ],
  source: "../docs/coexecution.md",
  bounds: [
    {
      variant: "baseline",
      role: "S1",
      gpu_work_ms: 5.326699666666666,
      eligible_linear_work_ms: 4.0788085,
      eligible_fraction: 0.7657290170730482,
      zero_target_cost_ceiling: 4.268561080446787,
      scenarios: [
        {
          assumed_local_speedup: 2,
          new_gpu_work_ms: 3.2872954166666664,
          gpu_work_speedup: 1.6203897099299842,
        },
        {
          assumed_local_speedup: 4,
          new_gpu_work_ms: 2.2675932916666666,
          gpu_work_speedup: 2.349054253354038,
        },
      ],
    },
    {
      variant: "baseline",
      role: "S2",
      gpu_work_ms: 9.1465965,
      eligible_linear_work_ms: 5.146185166666666,
      eligible_fraction: 0.5626338897388408,
      zero_target_cost_ceiling: 2.286414005426442,
      scenarios: [
        {
          assumed_local_speedup: 2,
          new_gpu_work_ms: 6.573503916666667,
          gpu_work_speedup: 1.3914339469410573,
        },
        {
          assumed_local_speedup: 4,
          new_gpu_work_ms: 5.286957624999999,
          gpu_work_speedup: 1.7300302269776562,
        },
      ],
    },
    {
      variant: "rope_rms",
      role: "S1",
      gpu_work_ms: 5.328166666666667,
      eligible_linear_work_ms: 4.080668833333333,
      eligible_fraction: 0.7658673402358533,
      zero_target_cost_ceiling: 4.271082902348392,
      scenarios: [
        {
          assumed_local_speedup: 2,
          new_gpu_work_ms: 3.2878322500000006,
          gpu_work_speedup: 1.6205713252756937,
        },
        {
          assumed_local_speedup: 4,
          new_gpu_work_ms: 2.2676650416666675,
          gpu_work_speedup: 2.349626849100527,
        },
      ],
    },
    {
      variant: "rope_rms",
      role: "S2",
      gpu_work_ms: 7.321785166666667,
      eligible_linear_work_ms: 5.1264935,
      eligible_fraction: 0.7001698879856507,
      zero_target_cost_ceiling: 3.3352220471843137,
      scenarios: [
        {
          assumed_local_speedup: 2,
          new_gpu_work_ms: 4.758538416666667,
          gpu_work_speedup: 1.5386626156095093,
        },
        {
          assumed_local_speedup: 4,
          new_gpu_work_ms: 3.4769150416666674,
          gpu_work_speedup: 2.1058280340255173,
        },
      ],
    },
  ],
};
