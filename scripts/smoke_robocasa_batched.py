#!/usr/bin/env python3
"""Start N real RoboCasa environments, batch observations, and step each one.
This is a plumbing smoke test, not a benchmark success-rate evaluation.
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def worker(rank, args, barrier, results):
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      TOKENIZERS_PARALLELISM='false', MUJOCO_GL=args.render_backend,
                      PYOPENGL_PLATFORM=args.render_backend, LP_NUM_THREADS='1')
    log_path = Path(args.output) / f'worker-{rank}.log'
    env = client = None
    started = time.monotonic()
    try:
        with log_path.open('w', buffering=1) as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            sys.path.insert(0, str(ROOT))
            import numpy as np
            import torch
            from eval_robocasa.main import (create_env, EvalClient, render_obs,
                                           camera_names, rgbs_to_pil_images, center_crop_pil)
            torch.set_num_threads(1)
            env = create_env(args.task, seed=rank)
            env.reset()
            controller = env.robots[0].composite_controller
            arm = controller.part_controllers[controller.arms[0]]
            base2world = np.eye(4)
            base2world[:3, :3] = arm.origin_ori
            base2world[:3, 3] = arm.origin_pos
            client = EvalClient(host=args.host, port=args.port, model_path=args.model)
            latencies, render_times, step_times = [], [], []
            initialized_seconds = time.monotonic() - started
            active_start = None
            for call_index in range(args.calls):
                start = time.monotonic()
                rgb, state, _ = render_obs(env, camera_names, base2world)
                images = center_crop_pil(rgbs_to_pil_images(rgb), .95)
                request = client.prepare_request(state, images, env.get_ep_meta()['lang'])
                render_times.append(time.monotonic() - start)
                print(f'Worker {rank}: observation prepared', flush=True)
                if call_index == 0:
                    barrier.wait(timeout=args.timeout)
                    active_start = time.monotonic()
                start = time.monotonic()
                actions = client.client(**request)[0, :, :7].float().numpy()
                latencies.append(time.monotonic() - start)
                if actions.shape != (10, 7) or not np.isfinite(actions).all():
                    raise AssertionError(f'Invalid action: {actions.shape}')
                start = time.monotonic()
                for action in actions:
                    padded = np.zeros(env.action_spec[0].shape)
                    padded[:7] = action
                    env.step(padded)
                step_times.append(time.monotonic() - start)
            results.put(dict(rank=rank, calls=args.calls, steps=args.calls * 10,
                             response_seconds=latencies, render_seconds=render_times,
                             step_seconds=step_times, initialized_seconds=initialized_seconds,
                             active_seconds=time.monotonic() - active_start))
    except BaseException:
        error = traceback.format_exc()
        results.put(dict(rank=rank, error=error))
        barrier.abort()
    finally:
        if client is not None:
            client.close()
        if env is not None:
            env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--calls', type=int, default=2)
    p.add_argument('--task', default='OpenDrawer')
    p.add_argument('--render-backend', choices=['egl', 'osmesa'], default='osmesa')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=10086)
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=False)
    ctx = mp.get_context('spawn')
    barrier, results = ctx.Barrier(args.workers), ctx.Queue()
    processes = [ctx.Process(target=worker, args=(rank, args, barrier, results))
                 for rank in range(args.workers)]
    rows = []
    start = time.monotonic()
    try:
        for proc in processes:
            proc.start()
        deadline = start + args.timeout
        while len(rows) < args.workers:
            row = results.get(timeout=max(.1, deadline - time.monotonic()))
            rows.append(row)
            if 'error' in row:
                raise RuntimeError(row['error'])
        for proc in processes:
            proc.join(timeout=30)
        if any(p.exitcode != 0 for p in processes):
            raise RuntimeError('Worker did not exit cleanly')
    finally:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
        report = dict(workers=args.workers, render_backend=args.render_backend, elapsed_seconds=time.monotonic() - start,
                      results=sorted(rows, key=lambda x: x['rank']))
        if len(rows) == args.workers and all('active_seconds' in r for r in rows):
            report['requests_per_second_after_initialization'] = (
                sum(r['calls'] for r in rows) / max(r['active_seconds'] for r in rows))
        (Path(args.output) / 'smoke.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
