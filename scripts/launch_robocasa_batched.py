#!/usr/bin/env python3
"""Run isolated RoboCasa processes against one shared inference endpoint."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

TASKS = ('PnPCounterToCab PnPCabToCounter PnPCounterToSink PnPSinkToCounter '
         'PnPCounterToMicrowave PnPMicrowaveToCounter PnPCounterToStove PnPStoveToCounter '
         'OpenSingleDoor CloseSingleDoor OpenDoubleDoor CloseDoubleDoor OpenDrawer CloseDrawer '
         'TurnOnStove TurnOffStove TurnOnSinkFaucet TurnOffSinkFaucet TurnSinkSpout '
         'CoffeeSetupMug CoffeeServeMug CoffeePressButton TurnOnMicrowave TurnOffMicrowave').split()
ROOT = Path(__file__).resolve().parents[1]


def make_jobs(tasks, episodes, shard_size, start_seed):
    return [(task, start_seed + offset, min(shard_size, episodes - offset))
            for offset in range(0, episodes, shard_size) for task in tasks]


async def run(args):
    semaphore = asyncio.Semaphore(args.workers)
    active = set()
    results = []
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
    (output / 'run.json').write_text(json.dumps(vars(args), indent=2))

    async def worker(task, seed, count):
        async with semaphore:
            shard = output / f'{task}_seed{seed}'
            shard.mkdir()
            command = [sys.executable, '-u', str(ROOT / 'eval_robocasa/main.py'),
                       '--task-name', task, '--model-path', args.model,
                       '--num-eval-episodes', str(count), '--start-seed', str(seed),
                       '--server-addr', args.host, '--server-port', str(args.port),
                       '--save-root-dir', str(shard)]
            if not args.video:
                command.append('--no-save-video')
            with (shard / 'worker.log').open('w') as log:
                proc = await asyncio.create_subprocess_exec(
                    *command, cwd=ROOT, env=env, stdout=log, stderr=asyncio.subprocess.STDOUT)
                active.add(proc)
                try:
                    code = await proc.wait()
                finally:
                    active.discard(proc)
            if code:
                raise RuntimeError(f'{task} seed {seed} failed ({code}); see {shard / "worker.log"}')
            results.append(json.loads((shard / f'eval_results_{task}.json').read_text()))
            print(f'Completed {task} seeds {seed}..{seed + count - 1}', flush=True)

    tasks = [asyncio.create_task(worker(*job)) for job in
             make_jobs(args.tasks, args.episodes, args.episodes_per_shard, args.start_seed)]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        # Stop active simulators on failure or Ctrl-C; do not leave orphan jobs.
        processes = list(active)
        for proc in processes:
            if proc.returncode is None:
                proc.terminate()
        if processes:
            try:
                await asyncio.wait_for(asyncio.gather(*(p.wait() for p in processes)), timeout=10)
            except asyncio.TimeoutError:
                for proc in processes:
                    if proc.returncode is None:
                        proc.kill()
                await asyncio.gather(*(p.wait() for p in processes))
        await asyncio.gather(*tasks, return_exceptions=True)
    summary = {}
    for result in results:
        for name, row in result.items():
            total = summary.setdefault(name, dict(num_success_rollouts=0, num_rollouts=0))
            for key in total:
                total[key] += row[key]
    for row in summary.values():
        row['success_rate'] = row['num_success_rollouts'] / row['num_rollouts']
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Results: {output / "summary.json"}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--host', default='localhost')
    p.add_argument('--port', type=int, default=10086)
    p.add_argument('--tasks', nargs='+', choices=TASKS, default=TASKS)
    p.add_argument('--episodes', type=int, default=100, help='Episodes per task')
    p.add_argument('--episodes-per-shard', type=int, default=5)
    p.add_argument('--start-seed', type=int, default=0)
    p.add_argument('--output', required=True, help='New directory; existing paths are rejected')
    p.add_argument('--video', action='store_true', help='Save videos (extra CPU, RAM, I/O)')
    args = p.parse_args()
    if min(args.workers, args.episodes, args.episodes_per_shard) <= 0:
        p.error('workers, episodes, and episodes-per-shard must be positive')
    if len(set(args.tasks)) != len(args.tasks):
        p.error('tasks must be unique')
    return args


if __name__ == '__main__':
    asyncio.run(run(parse_args()))
