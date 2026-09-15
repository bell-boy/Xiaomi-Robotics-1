# XR-1 / RoboCasa batch-window tuning — 2026-09-15

## Decision

Use **20 ms**, maximum batch **16**, one XR-1 instance, and **32 independently running OSMesa environments**. It won both repeats and the pooled steady-state score: **176.33 simulation steps/s (17.64 requests/s)**, 3.25% above 50 ms and 5.83% above 100 ms. The fixed-work completion metric also selects 20 ms. This is a measured choice among these three windows on this workload, not a universal optimum.

## Method

- Existing RTX 5880 Ada 48 GB box; 30.72 effective CPU cores; 342 GiB enforced RAM limit. GPU power limit 285 W. No hardware or model changes between trials.
- Model revision `82b7c2210c1a39fca4a87e681cffbbed134ccde0`; harness commit `1e4b5c741b14c7204fdbc759e4fd1874282c6f08`.
- Run order: 20, 100, 50, 50, 100, 20 ms. Two repeats each to expose time/order variability. Each run reloads exactly one model, maximum batch 16.
- OpenDrawer seeds 0–31. Each environment performs 20 warmup calls and 160 measured calls, ten physics steps per call. Across all six runs: 30,720 measured requests and 307,200 measured simulation steps, plus warmup.
- First run records an action tape. Later runs still render, preprocess, request real XR-1 inference, receive and decode outputs, then apply the recorded action chunk. Every input is verified against an exact SHA-256 fingerprint of tensors and scalar fields. All six runs passed workload verification.
- One barrier after initialization; no barrier after warmup or between requests. Environments remain independent. Replaying actions isolates scheduler changes from numerical differences in batched outputs.
- Primary score counts actual physics-step completions in the common steady interval: after every worker has finished its own warmup, before the first worker finishes its measured work. This keeps all 32 workers active; intervals are 253–295 seconds. Pool counts over summed durations.
- Also report fixed-work throughput over the full measured span, including asynchronous start and drain. The 100 ms runs have substantially larger completion spread; this metric penalizes that tail.
- Server events capture actual batches, queue waits/depth (200 ms samples), dispatch gaps, transfer timings and CUDA-event stage times. NVML GPU/CPU-cgroup telemetry is sampled every 200 ms. Input fingerprinting is included in all runs (~5.6 ms/call). Profiling instrumentation is identical across trials.
- This is a fixed-length throughput trace, continuing beyond task success; it does not estimate task success rates. The prior short 15.83 requests/s check uses a different warmup/timing method and is not a controlled before/after comparison.

## Throughput

| Wait | Repeat 1 steps/s | Repeat 2 steps/s | Pooled steps/s | Requests/s | Fixed-work steps/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 ms | 174.79 | 177.90 | 176.33 | 17.64 | 173.94 |
| 50 ms | 168.61 | 172.97 | 170.79 | 17.08 | 168.29 |
| 100 ms | 166.91 | 166.34 | 166.62 | 16.67 | 142.40 |

Both 20 ms repeats exceed both 50 ms repeats. The 50 ms repeats vary by 2.6%; use the measured ordering without claiming a formal confidence interval from only two runs.

## Batches, latency, queue and GPU

| Trial | Mean batch | Full 16 (%) | Request p50 / p95 (ms) | Queue mean / p95 | GPU util mean (%) | Samples <5% util (%) | Worker busy (%) | Collection (%) | Mean dispatch gap (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01-wait-20 | 10.67 | 1.48 | 1000 / 1120 | 5.28 / 15 | 57.22 | 0.35 | 96.36 | 3.25 | 22.21 |
| 02-wait-100 | 13.88 | 27.96 | 984 / 1606 | 3.79 / 12 | 53.30 | 11.11 | 89.03 | 10.54 | 91.09 |
| 03-wait-50 | 10.70 | 1.30 | 1059 / 1208 | 6.29 / 12 | 53.82 | 2.32 | 91.78 | 7.81 | 52.18 |
| 04-wait-50 | 10.69 | 33.62 | 1016 / 1142 | 5.45 / 16 | 58.08 | 1.03 | 94.12 | 5.40 | 36.37 |
| 05-wait-100 | 13.08 | 23.17 | 1022 / 1583 | 4.28 / 12 | 52.28 | 10.10 | 88.56 | 11.04 | 89.92 |
| 06-wait-20 | 10.69 | 0.21 | 964 / 1096 | 5.56 / 11 | 59.02 | 0.35 | 96.19 | 3.35 | 22.89 |

Actual batch-size histograms below count completed batches fully inside each steady interval. Similar means can hide very different distributions (notably the two 50 ms runs).

| Trial | Batch size: count |
| --- | --- |
| 01-wait-20 | 5: 75, 6: 60, 7: 16, 8: 13, 9: 1, 10: 14, 11: 59, 12: 50, 13: 27, 14: 27, 15: 123, 16: 7 |
| 02-wait-100 | 6: 2, 7: 2, 8: 4, 9: 3, 10: 14, 11: 15, 12: 26, 13: 40, 14: 56, 15: 57, 16: 85 |
| 03-wait-50 | 5: 2, 6: 2, 7: 5, 8: 44, 9: 64, 10: 94, 11: 129, 12: 56, 13: 14, 14: 43, 15: 4, 16: 6 |
| 04-wait-50 | 4: 1, 5: 2, 6: 118, 7: 13, 8: 46, 9: 13, 10: 118, 11: 2, 12: 1, 16: 159 |
| 05-wait-100 | 5: 1, 6: 5, 7: 14, 8: 11, 9: 16, 10: 16, 11: 18, 12: 32, 13: 29, 14: 59, 15: 51, 16: 76 |
| 06-wait-20 | 7: 23, 8: 2, 9: 95, 10: 40, 11: 192, 12: 93, 14: 22, 15: 4, 16: 1 |

`summary.json` includes p99/max latency, queue wait distributions, per-size service times, sampled low-utilization gap durations, GPU clocks/power/temperature, client stages and 30-second throughput blocks. NVML utilization is a sampled activity indicator, not SM occupancy. Dispatch gaps measure time between policy calls; they do not measure idle time inside a model call.

## Why inference-only is faster

1. **Partial batches and the model service curve dominate the observed gap.** At 20 ms the mean actual batch is 10.67–10.69; inference-only service during those batches is 18.13–18.49 requests/s. Multiplying by the 96.2–96.4% worker duty cycle nearly explains the measured 17.5–17.8 requests/s. Cached real observations, verified against the same tape, reach 3.62, 18.03 and **21.90 requests/s** at fixed batches 1, 8 and 16. Thus the old synthetic ~22 requests/s ceiling is reproducible with real inputs, but not with the live batch mixture. This is evidence consistent with partial-batch efficiency, not an exact causal allocation of every millisecond.
2. **Longer waits hurt overlap.** The 100 ms window increases mean batch to 13.08–13.88, but busy-service rate only reaches 18.76–18.78 requests/s. Collection consumes 10.5–11.0% of the interval, compared with 3.2–3.3% at 20 ms. Request p95 grows to 1.58–1.61 s from 1.10–1.12 s. At 20 ms, non-collection dispatch gaps account for under 0.5% of time: the server is rarely starved for long.
3. **CPU rendering/physics affect arrivals, but aggregate CPU capacity is not exhausted.** At 20 ms, rendering takes 403–406 ms/cycle, physics 429–431 ms, preprocessing 18.5–18.7 ms, and request waiting about 0.94–0.96 s. The 32 independent clients overlap this work. CPU utilization is 54–55% of the 30.72-core quota, with zero steady-state throttled periods. Across all windows it is 51–55%. This does not rule out single-thread or memory-bandwidth limits, and changing rendering would require its own matched experiment.
4. **Transfers are secondary.** At 20 ms CPU collation is 16–19 ms/batch, host-to-device stream time ~14.5–14.9 ms, and device-to-host ~0.2 ms, versus ~578–588 ms total inference. Client serialization and sending each take ~2.6–2.7 ms; input payload is ~4.73 MB/request. Transfer work is measurable but cannot account for most of the throughput difference. Host/device collation and copies are not overlapped with the next model batch in this implementation.
5. **Inference has internal CPU/GPU dispatch overhead.** The separately profiled three batch-16 calls show ~49,155 CUDA kernel launches, substantial command-buffer/launch overhead and stream synchronization. The trace supports an internal execution-overhead contribution despite high server-worker duty cycle. Profiling increased batch duration to ~0.92 s, so its percentages are diagnostic only; they are not used to choose the window. CUDA-event stage times include host dispatch gaps. No claim of pure GPU compute saturation follows from them.

The isolated diagnostic uses the 32 initial real observations, five warmup batches and 40 timed batches at each size, with no simulators running. It is a diagnostic subset, not a substitute for the long end-to-end trace. Per-call NVML samples in that diagnostic are not time-weighted utilization averages.

## Memory and operation

GPU memory peaks around **13.03 GiB** in these runs. Sampled container RAM peaks at **261.46 GiB**, above the earlier short-check value (~226 GiB). The existing 342 GiB container fits; provision at least **300 GiB** for reproducing these 32 OSMesa environments and verify actual limits. Other task assets can change requirements. Maximum batch remains 16.

Normal serving uses the selected 20 ms default; JSONL profiling stays off unless explicitly enabled. The 32-worker launcher retains OSMesa and independent workers.

## Reproduction

Use the dependency versions in `docs/BATCHED_ROBOCASA.md`. Stop the normal server before running the experiment; the harness owns the only model instance. Install `nvidia-ml-py` in the deployment environment for telemetry.

```bash
python scripts/tune_batch_wait.py --model /workspace/checkpoints/Xiaomi-Robotics-1-RoboCasa \
  --workers 32 --warmup 20 --calls 160 --windows 20 100 50 50 100 20 \
  --output /workspace/new-batch-wait-tuning
python scripts/analyze_batch_wait.py /workspace/new-batch-wait-tuning --output /workspace/new-analysis
```

To reanalyze the saved experiment, extract `raw-trials.tar.gz` then pass the extracted `batch-wait-tuning` directory to `analyze_batch_wait.py`. The archive contains raw client/server/hardware events, configs, completion markers and action tapes.

```bash
/opt/conda/envs/robocasa/bin/python scripts/diagnose_cached_inference.py capture \
  --model /workspace/checkpoints/Xiaomi-Robotics-1-RoboCasa \
  --tapes /workspace/new-batch-wait-tuning/tapes --output /workspace/new-cached-inputs
python scripts/diagnose_cached_inference.py measure \
  --model /workspace/checkpoints/Xiaomi-Robotics-1-RoboCasa \
  --inputs /workspace/new-cached-inputs --output /workspace/new-cached-diagnostic --profile
```

All output directories for experiment/diagnostic runs must be new. Cached tensor files are regenerated from the saved seeds/tapes rather than committed. `cached-inference.json`, `operator-times.txt` and `cuda-trace.json.gz` preserve the diagnostic results.
