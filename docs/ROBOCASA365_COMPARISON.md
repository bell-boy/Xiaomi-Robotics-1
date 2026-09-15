# Four-GPU RoboCasa365 comparison

This evaluation compares the released RoboCasa365-trained checkpoint with general
XR-1-5B inference weights, without performing fine-tuning. The base weights need
an explicit RoboCasa365 input/output adapter; the comparison must disclose use
of the trained checkpoint's processor, action mask and normalization statistics.
It is an adapter-based zero-shot baseline, not a native published base benchmark.

## Protocol

- RoboCasa v1.0.1 commit `4f8a2980def75a55dff96b990745b83540425f09`.
- `target` split, `target50`: 18 atomic-seen, 16 composite-seen, 16 composite-unseen tasks.
- 50 episodes per task per policy, matching episode seeds starting at 7.
- Current official horizons; observation history 4, interval 2, 16 actions/query,
  center crop 0.95. No shortened horizons in the full evaluation.
- Four GPU servers, one model instance per GPU, maximum batch 16, wait 20 ms.
- Independent environments sharing a rollout queue. Initial concurrency is eight
  OSMesa workers per GPU; validate host memory and throughput before changing it.
- Per-request seeded noise depends on episode seed and request index, never GPU
  assignment or batch ordering. This differs from the original persistent
  server-global RNG stream, and is used for both policies.
- Videos stream to disk as three-camera MP4s. Stride 2, 20 FPS preserves the
  reference evaluator defaults; playback is faster than real time at 20 Hz control.

The prior RoboCasa v0.2 tuning is not a RoboCasa365 throughput measurement.
RoboCasa365 has video-history inputs, a different action representation and
renders observations each simulation step. Recheck memory and throughput.

## Checkpoints

- Trained: `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365`, revision
  `3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4`.
- Base: `XiaomiRobotics/Xiaomi-Robotics-1-5B`, revision
  `ee21d524b5c52ac961d941e1bc7d6d92836c3d5e`.

`convert_base_robocasa365.py` packages base weights with the task adapter. It
requires every inference key and shape to match, checks tensor equality, and
allows omitting only the three training-only choice heads. It preserves the
base runtime's +10 action-position offset. Its conversion report records source
SHA-256, omitted keys, parameter count and zero training steps. Do not treat a
failed conversion as permission to fill missing inference weights from the
fine-tuned checkpoint.

## Launch

The deployment and simulator environments are separate. The simulator needs the
current RoboCasa/robosuite versions; see the official installation guide. Use
Transformers 4.57.1 and Flash Attention 2 for deployment.

```bash
python scripts/launch_robocasa365_batched.py \
  --model /workspace/checkpoints/Xiaomi-Robotics-1-RoboCasa365 \
  --output /workspace/evaluations/robocasa365-trained \
  --gpus 0 1 2 3 --workers-per-gpu 8 --max-batch-size 16 --batch-wait-ms 20

python scripts/convert_base_robocasa365.py \
  --base-weights /workspace/checkpoints/Xiaomi-Robotics-1-5B/model_states.pt \
  --adapter /workspace/checkpoints/Xiaomi-Robotics-1-RoboCasa365 \
  --output /workspace/checkpoints/XR1-5B-zero-shot-365-adapter

python scripts/launch_robocasa365_batched.py \
  --model /workspace/checkpoints/XR1-5B-zero-shot-365-adapter \
  --output /workspace/evaluations/base-5b-no-finetuning \
  --gpus 0 1 2 3 --workers-per-gpu 8 --max-batch-size 16 --batch-wait-ms 20
```

Run sequentially so each evaluation uses all four GPUs. The launcher starts and
stops its own four servers, requires a new output directory, and stops workers
if a server or worker fails. It writes DONE only after all rollout results merge.
Failed runs retain scheduler errors/logs and partial videos for diagnosis.

For a four-GPU smoke test, add `--num-trials 4 --task-name CloseBlenderLid --horizon 32`.
Set `XR1_CAPTURE_REQUEST_DIR` to retain each episode's initial request for the
sampler consistency test. `check_robocasa365_batching.py` checks the trained
checkpoint's singleton forward and a mixed batch; run with no other model on
that GPU. Captured tensors are diagnostics, not committed evaluation data.

## Results and videos

```bash
python scripts/summarize_robocasa365_comparison.py \
  --trained /workspace/evaluations/robocasa365-trained \
  --base /workspace/evaluations/base-5b-no-finetuning \
  --output /workspace/evaluations/comparison
```

The summary refuses incomplete runs, missing/duplicate episodes, unpaired seeds
and missing videos. It writes SUMMARY.md, tasks.csv and comparison.json, including
video checksums. The full data must be preserved before stopping storage or
removing temporary upload copies. Compare these results with published numbers
only after matching task split, horizon version, renderer and sampling protocol.
