# One XR-1 GPU server, 32 parallel RoboCasa environments

This branch adds dynamic batching for the **RoboCasa v0.2** checkpoint and
`robocasa_mg` observations. One model stays on one GPU. Each environment owns
its simulator state and persistent socket; a single inference worker combines
queued observations, runs one batched VLM/action-head pass, then returns each
sample to its originating client.

## Run on a prepared machine

From the repository root, in the deployment environment:

```bash
export MODEL_PATH=/workspace/checkpoints/Xiaomi-Robotics-1-RoboCasa
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python -u deploy/server.py \
  --model "$MODEL_PATH" --host 127.0.0.1 --port 10086 \
  --max-batch-size 32 --batch-wait-ms 20
```

In the RoboCasa Python 3.10 environment:

```bash
python scripts/launch_robocasa_batched.py \
  --model "$MODEL_PATH" --workers 32 --host 127.0.0.1 --port 10086 \
  --episodes 100 --episodes-per-shard 5 --output /workspace/runs/robocasa-001
```

The launcher schedules episode shards across 32 processes, all using **the same
port**. It runs all 24 tasks by default. Use `--tasks OpenDrawer CloseDrawer`
for a subset. Each task's seeds are partitioned exactly once, and every shard
gets its own output directory. `summary.json` combines counts only after all
workers succeed. Failed workers stop the run and keep their logs. Existing
output directories are rejected to avoid mixing results from separate runs.
For 32 workers on a single task, use enough episodes/shards (e.g. 100 episodes
with `--episodes-per-shard 3`).

CPU thread pools are limited to one thread per simulator. Offscreen rendering
uses EGL on the shared GPU. Videos are disabled by this launcher; use `--video`
to enable them. The original evaluation entry point still saves videos by default.

Do not use the original multi-port launcher for this setup. The batched policy
currently validates the RoboCasa three-camera schema; other embodiments and
video inputs require their own collation support.

## Batching behavior

- Maximum batch: 32 by default; configurable with `--max-batch-size`.
- Wait window: 10 ms by default (20 ms in the command above). After the first
  request, collect up to the limit or until the window expires. Drain queued
  work immediately. Partial batches run without waiting for absent environments.
- At most 256 queued requests and 128 connected clients by default. Both limits
  are configurable. One outstanding request per connection preserves ordering.
- Text lengths can differ: left padding and attention masks preserve image/text
  alignment. Flattened image patches and image grids concatenate in request order.
- Each request gets its own seeded action noise, independent of batch membership
  and arrival order. The sampler retains the published five Euler steps. As with
  other BF16 batched execution, floating-point rounding can change outputs slightly.
- A request/inference error returns a structured error to the client. Timeouts
  cancel queued work; work already running on CUDA finishes in the sole worker.
- Logs include actual batch size, inference time, and queue depth. 32 concurrent
  environments does not guarantee every batch has 32 samples: simulation timing,
  rendering, and the wait window determine achieved batch sizes.

The socket protocol uses pickle and must only accept trusted clients. Keep it
bound to loopback. For remote simulation, use an SSH tunnel rather than exposing
port 10086 publicly. No inference port needs to be published on Vast.

## Validation

```bash
python -m unittest discover -s tests -v
python scripts/benchmark_batched.py --model "$MODEL_PATH" \
  --output /workspace/benchmark.json
# Run using RoboCasa's Python environment while the server is running:
python scripts/smoke_robocasa_batched.py --model "$MODEL_PATH" \
  --workers 32 --calls 2 --output /workspace/runs/smoke-001
```

The benchmark checks singleton agreement with the original checkpoint forward,
compares mixed-length batched observations, and measures batches 1/8/16/32.
Synthetic image measurements are inference throughput, not simulator throughput
or benchmark success rates. The smoke test creates real environments, renders
all three cameras, sends synchronized requests, checks action shapes/finiteness,
and applies returned actions separately to every simulator.

## Environment compatibility

Inference: PyTorch 2.8.0, torchvision 0.23.0, Transformers **4.57.1**, Flash
Attention 2.8.3, CUDA 12.8. The tested Vast image is
`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` (Python 3.11).

Simulation: Python **3.10**, RoboCasa **v0.2**, RoboSuite **v1.5.1**, MuJoCo
**3.2.6**, NumPy **1.23.3**, Numba **0.56.4**, OpenCV **4.8.1.78**, Mink **0.0.5**.
RoboSuite's current main branch requires a newer MuJoCo than RoboCasa v0.2 pins.
CPU-only PyTorch 2.8.0/torchvision 0.23.0 suffice for the simulation clients;
they still use the NVIDIA driver for EGL rendering.
