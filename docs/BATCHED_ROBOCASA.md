# One XR-1 GPU server, 32 parallel RoboCasa environments

The tested configuration uses **32 independent RoboCasa v0.2 environments with
CPU rendering**, connected to **one XR-1 GPU instance with maximum batch 16**.
This avoids competition between kitchen renderers and model inference for GPU
memory. The shared server also supports larger batches and more clients.

## Start the server and evaluation

From the repository root, in the deployment environment:

```bash
export MODEL_PATH=/workspace/checkpoints/Xiaomi-Robotics-1-RoboCasa
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python -u deploy/server.py \
  --model "$MODEL_PATH" --host 127.0.0.1 --port 10086 \
  --max-batch-size 16 --batch-wait-ms 20
```

In the RoboCasa Python 3.10 environment:

```bash
python scripts/launch_robocasa_batched.py \
  --model "$MODEL_PATH" --workers 32 --render-backend osmesa \
  --host 127.0.0.1 --port 10086 --episodes 100 --episodes-per-shard 5 \
  --output /workspace/runs/robocasa-001
```

All workers use **the same port**. The launcher runs all 24 tasks by default;
use `--tasks OpenDrawer CloseDrawer` for a subset. It divides each task's seeds
into nonoverlapping shards, gives every shard its own output directory, and
combines result counts into `summary.json` only after every worker succeeds.
Existing output directories are rejected. A failed worker stops the run and
preserves logs. To fill 32 workers with a single task, use enough shards, e.g.
100 episodes and `--episodes-per-shard 3`.

Videos are disabled by the launcher to save CPU, RAM, and I/O; enable with
`--video`. The original evaluator still saves videos by default. CPU thread
pools and OSMesa's rendering threads are limited to one thread per simulator.

Do not use the original multi-port launcher for this configuration.

## Rendering and memory

`--render-backend osmesa` (the new launcher's default) renders on the CPU and
reserves GPU memory for inference. `--render-backend egl` uses the GPU. Both
retain the original image resolution and quality settings; different renderer
implementations can produce pixel differences, so evaluate success rates for
your chosen backend.

On the tested 48 GB RTX 5880 Ada, 32 simultaneous EGL environments crowded out
inference during initialization; even 16 EGL environments produced a CUDA OOM.
Eight EGL environments worked. Reducing only the inference batch size cannot
free renderer memory: reduce the number of EGL workers or use CPU rendering.

CPU rendering uses substantial host RAM. The 32-environment run had a sampled
container memory usage of about **226 GiB**; its enforced memory limit was
**342 GiB**. For recreation, provision at least 256 GiB and check the actual
container limit. Memory use depends on kitchen/task assets; monitor it when
changing tasks or concurrency.

## Measured validation

RTX 5880 Ada, 48 GB, one shared model:

| Configuration | Requests/second |
| --- | ---: |
| Synthetic inference only, batch 1 | 3.71 |
| Synthetic inference only, batch 8 | 18.15 |
| Synthetic inference only, batch 16 | 22.03 |
| Synthetic inference only, batch 32 | 24.00 |
| 8 real environments, EGL, maximum batch 16 | 7.07 |
| 16 real environments, OSMesa, maximum batch 16 | 8.69 |
| 32 real environments, OSMesa, maximum batch 16 | **15.83** |

Real-environment measurements used OpenDrawer seeds 0..N-1, 10 calls per
environment, and 10 simulation steps per returned action chunk. Their timing
starts after all environments prepare the initial observation, includes later
rendering/preprocessing, socket inference, and stepping, and ends when the
slowest worker finishes. These are short-run throughput checks, not full
benchmark success-rate measurements. The 32-environment run completed all
**320 requests and 3,200 simulation steps** without errors.

Synthetic inference used three-camera inputs with different instruction lengths.
Batch 32 peaked at 13.05 GiB of PyTorch allocations. The new singleton sampler
matched the original checkpoint output exactly in the test; mixed-length batch
comparisons differed by at most 0.0391 in normalized actions (BF16 rounding).

## Batching behavior

- One inference thread owns the model. Multiple client connections enqueue one
  outstanding observation each; results return to their originating connection.
- Maximum batch defaults to 16. The wait window defaults to 10 ms; the commands
  above use 20 ms. Partial batches run when the window expires. No environment
  must wait for every other environment to issue a request.
- Text is left-padded with matching attention masks; flattened camera patches
  and image grids concatenate in request order. Action/state tensors batch on
  their leading dimension. Each output retains a singleton batch dimension.
- Every request gets independent seeded action noise, unaffected by queue order
  or batch membership. The sampler retains the published five Euler steps.
- Default limits: 256 queued requests, 128 clients, 300-second request timeout.
  Queue limits/timeouts return errors; queued timed-out work is cancelled. CUDA
  work already running finishes in the sole inference thread.
- Logs report actual batch sizes, inference time, and queue depth. Environment
  count and maximum batch size are separate controls; achieved batches vary.

The batched policy validates the three-camera `robocasa_mg` schema. Other
embodiments and video inputs require additional collation support. The legacy
pickle socket protocol requires trusted clients: keep the server bound to
loopback and use SSH forwarding for remote clients. Do not expose port 10086
publicly.

## Installation and tests

For a fresh Vast container, use image
`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` and run:

```bash
bash scripts/setup_vast_robocasa.sh
```

The script installs both environments, pins checkpoint revision
`82b7c2210c1a39fca4a87e681cffbbed134ccde0`, and downloads kitchen assets. It
expects new dependency checkout paths. To support optional EGL, create the
container with `NVIDIA_DRIVER_CAPABILITIES=all`.

Inference dependencies: PyTorch 2.8.0, torchvision 0.23.0, Transformers 4.57.1,
Flash Attention 2.8.3, CUDA 12.8, Python 3.11.

Simulation: Python 3.10, CPU PyTorch 2.8.0/torchvision 0.23.0, RoboCasa v0.2,
RoboSuite v1.5.1, MuJoCo 3.2.6, NumPy 1.23.3, Numba 0.56.4, OpenCV 4.8.1.78,
Mink 0.0.5. Current RoboSuite main conflicts with RoboCasa v0.2's MuJoCo pin.
The evaluator includes a v1.5.1 fallback reading the same qpos entries as the
newer joint-state getter methods.

```bash
# Deployment environment:
python -m unittest discover -s tests -v
python scripts/benchmark_batched.py --model "$MODEL_PATH" \
  --output /workspace/benchmark.json
# RoboCasa environment, with the server already running:
python scripts/smoke_robocasa_batched.py --model "$MODEL_PATH" \
  --workers 32 --calls 10 --render-backend osmesa \
  --output /workspace/runs/smoke-001
```
