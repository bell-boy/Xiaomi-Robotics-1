#!/usr/bin/env python3
"""Matched live-simulator workload for independent-client batching comparisons.

Record an action tape once, then replay those actions (not new policy outputs)
while still rendering, preprocessing, requesting/decoding real model inference,
and stepping the simulator. Exact input fingerprints verify matched workloads.
Only initialization is synchronized; no per-call or post-warmup barriers.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import signal
import struct
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(request):
    digest = hashlib.sha256()
    for key, value in sorted(request.items()):
        digest.update(key.encode())
        if hasattr(value, 'numpy'):
            import torch
            digest.update(str((str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def timed_call(client, request):
    start = time.monotonic()
    payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
    serialized = time.monotonic()
    client.client_socket.sendall(struct.pack('>I', len(payload)) + payload)
    sent = time.monotonic()
    length = struct.unpack('>I', client._recv_all(4))[0]
    if not 0 < length <= 64 * 1024 * 1024:
        raise RuntimeError('Invalid response length')
    received = client._recv_all(length)
    received_at = time.monotonic()
    response = pickle.loads(received)
    unpacked = time.monotonic()
    if isinstance(response, dict) and 'error' in response:
        raise RuntimeError(response['error'])
    actions = client.processor.decode_action(response, robot_type=request['task_id'])[0, :, :7].float().numpy()
    decoded = time.monotonic()
    return actions, dict(request_start=start, response_end=decoded,
                         serialize_ms=(serialized-start)*1000, send_ms=(sent-serialized)*1000,
                         response_wait_ms=(received_at-sent)*1000,
                         unpickle_ms=(unpacked-received_at)*1000,
                         decode_ms=(decoded-unpacked)*1000, input_bytes=len(payload), output_bytes=length)


def worker(rank, args, barrier, results):
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      TOKENIZERS_PARALLELISM='false', MUJOCO_GL='osmesa',
                      PYOPENGL_PLATFORM='osmesa', LP_NUM_THREADS='1')
    env = client = None
    rows, actions_tape, hashes = [], [], []
    output = Path(args.output)
    try:
        with (output / f'worker-{rank}.log').open('w', buffering=1) as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            sys.path.insert(0, str(ROOT))
            import numpy as np
            import torch
            from eval_robocasa.main import (create_env, EvalClient, render_obs,
                                           camera_names, rgbs_to_pil_images, center_crop_pil)
            torch.set_num_threads(1)
            random.seed(rank)
            np.random.seed(rank)
            torch.manual_seed(rank)
            reference = None
            tape_path = Path(args.tape_dir) / f'worker-{rank}.npz'
            if args.mode == 'replay':
                reference = np.load(tape_path, allow_pickle=False)
                if len(reference['actions']) != args.warmup + args.calls:
                    raise RuntimeError('Action-tape length differs from requested workload')
            env = create_env(args.task, seed=rank)
            env.reset()
            controller = env.robots[0].composite_controller
            arm = controller.part_controllers[controller.arms[0]]
            base2world = np.eye(4)
            base2world[:3, :3], base2world[:3, 3] = arm.origin_ori, arm.origin_pos
            client = EvalClient(host=args.host, port=args.port, model_path=args.model)
            # The sole barrier is before any warmup work starts.
            barrier.wait(timeout=args.timeout)
            for index in range(args.warmup + args.calls):
                cpu_start, start = time.process_time(), time.monotonic()
                rgb, state, _ = render_obs(env, camera_names, base2world)
                rendered = time.monotonic()
                images = center_crop_pil(rgbs_to_pil_images(rgb), .95)
                request = client.prepare_request(state, images, env.get_ep_meta()['lang'])
                prepared = time.monotonic()
                signature = fingerprint(request)
                hashed = time.monotonic()
                if reference is not None and signature != reference['fingerprints'][index]:
                    raise RuntimeError(f'Workload mismatch rank={rank} call={index}: {signature} != {reference["fingerprints"][index]}')
                returned, timing = timed_call(client.client, request)
                if returned.shape != (10, 7) or not np.isfinite(returned).all():
                    raise RuntimeError('Invalid model output')
                actions = returned if reference is None else reference['actions'][index]
                step_start = time.monotonic()
                step_ends = []
                for action in actions:
                    padded = np.zeros(env.action_spec[0].shape)
                    padded[:7] = action
                    env.step(padded)
                    step_ends.append(time.monotonic())
                end = time.monotonic()
                rows.append(dict(rank=rank, index=index, warmup=index < args.warmup,
                                 cycle_start=start, cycle_end=end, step_ends=step_ends,
                                 render_ms=(rendered-start)*1000,
                                 preprocess_ms=(prepared-rendered)*1000,
                                 fingerprint_ms=(hashed-prepared)*1000,
                                 physics_ms=(end-step_start)*1000,
                                 cpu_ms=(time.process_time()-cpu_start)*1000, **timing))
                hashes.append(signature)
                if reference is None:
                    actions_tape.append(actions.copy())
                if index % 20 == 0:
                    print(f'worker={rank} call={index} phase={"warmup" if index < args.warmup else "measure"}', flush=True)
            if reference is None:
                tape_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(tape_path, actions=np.stack(actions_tape), fingerprints=np.array(hashes))
            with (output / f'worker-{rank}.json').open('w') as f:
                json.dump(rows, f, separators=(',', ':'))
            results.put(dict(rank=rank, calls=len(rows), workload_verified=True))
    except BaseException:
        results.put(dict(rank=rank, error=traceback.format_exc()))
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
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--calls', type=int, default=160, help='Measured calls per worker after warmup')
    p.add_argument('--task', default='OpenDrawer')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=10086)
    p.add_argument('--timeout', type=int, default=3600)
    p.add_argument('--mode', choices=['record', 'replay'], required=True)
    p.add_argument('--tape-dir', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if min(args.workers, args.calls, args.warmup) <= 0:
        p.error('workers, calls, and warmup must be positive')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
    def interrupt(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupt)
    ctx = mp.get_context('spawn')
    barrier, results = ctx.Barrier(args.workers), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(rank, args, barrier, results)) for rank in range(args.workers)]
    completed = []
    try:
        for proc in procs:
            proc.start()
        deadline = time.monotonic() + args.timeout
        while len(completed) < args.workers:
            row = results.get(timeout=max(.1, deadline-time.monotonic()))
            completed.append(row)
            if 'error' in row:
                raise RuntimeError(row['error'])
        for proc in procs:
            proc.join(timeout=30)
        if any(proc.exitcode != 0 for proc in procs):
            raise RuntimeError('Worker did not exit cleanly')
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
        (output / 'completion.json').write_text(json.dumps(completed, indent=2))
    print(f'Completed {args.workers} workers, {args.warmup} warmup + {args.calls} measured calls each')


if __name__ == '__main__':
    main()
