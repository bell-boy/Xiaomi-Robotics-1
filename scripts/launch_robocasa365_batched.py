#!/usr/bin/env python3
"""Run independent RoboCasa365 clients against one batched model per GPU."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sim-python', default='/opt/conda/envs/robocasa365/bin/python')
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    p.add_argument('--workers-per-gpu', type=int, default=8)
    p.add_argument('--max-batch-size', type=int, default=16)
    p.add_argument('--batch-wait-ms', type=float, default=20)
    p.add_argument('--base-port', type=int, default=10086)
    p.add_argument('--num-trials', type=int, default=50)
    p.add_argument('--split', choices=['pretrain', 'target'], default='target')
    p.add_argument('--task-name', action='append')
    p.add_argument('--horizon', type=int)
    args = p.parse_args()
    if min(args.workers_per_gpu, args.max_batch_size, args.num_trials) < 1 or len(set(args.gpus)) != len(args.gpus):
        p.error('Positive counts and distinct GPUs required')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'logs').mkdir()
    queue = args.output/'scheduler'
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa', LP_NUM_THREADS='1')
    command = [args.sim_python, '-u', 'eval_robocasa365/dynamic_eval.py', 'init', '--queue-dir', str(queue), '--',
               '--model-path', args.model, '--save-root-dir', str(args.output), '--run-id', 'results',
               '--split', args.split, '--task-set', 'target50', '--num-trials', str(args.num_trials),
               '--replan-steps', '16', '--obs-history', '4', '--obs-interval', '2', '--seed', '7',
               '--crop-ratio', '0.95', '--save-videos', '--video-stride', '2', '--video-fps', '20']
    for task in args.task_name or []:
        command += ['--task-name', task]
    if args.horizon is not None:
        command += ['--horizon', str(args.horizon)]
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    (args.output/'run-config.json').write_text(json.dumps(vars(args), default=str, indent=2))
    servers, workers = [], []
    def interrupted(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for index, gpu in enumerate(args.gpus):
            port = args.base_port+index
            gpu_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4')
            with (args.output/f'logs/server-{gpu}.log').open('w') as log:
                proc = subprocess.Popen([sys.executable, '-u', 'deploy/server.py', '--model', args.model,
                    '--host', '127.0.0.1', '--port', str(port), '--max-batch-size', str(args.max_batch_size),
                    '--batch-wait-ms', str(args.batch_wait_ms)], cwd=ROOT, env=gpu_env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            servers.append(proc)
            deadline = time.monotonic()+240
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'GPU {gpu} server failed')
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=1):
                        break
                except OSError:
                    if time.monotonic()>deadline:
                        raise TimeoutError(f'Server {gpu} startup timed out')
                    time.sleep(.5)
        for index, gpu in enumerate(args.gpus):
            for client in range(args.workers_per_gpu):
                worker_id = f'gpu-{gpu}-client-{client}'
                with (queue/f'logs/{worker_id}.log').open('w') as log:
                    workers.append(subprocess.Popen([args.sim_python, '-u', 'eval_robocasa365/dynamic_eval.py',
                        'worker', '--queue-dir', str(queue), '--worker-id', worker_id,
                        '--server-addr', '127.0.0.1', '--server-port', str(args.base_port+index)],
                        cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=''), stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True))
        while any(p.poll() is None for p in workers):
            if any(p.poll() not in (None, 0) for p in workers) or any(p.poll() is not None for p in servers):
                raise RuntimeError('Worker or server failed; inspect logs and scheduler errors')
            time.sleep(2)
        if any(p.returncode for p in workers):
            raise RuntimeError('A worker failed')
        subprocess.run([args.sim_python, 'eval_robocasa365/dynamic_eval.py', 'merge', '--queue-dir', str(queue)],
                       cwd=ROOT, env=env, check=True)
        (args.output/'DONE').write_text('All episodes merged successfully\n')
    finally:
        for proc in workers+servers:
            stop(proc)


if __name__ == '__main__':
    main()
