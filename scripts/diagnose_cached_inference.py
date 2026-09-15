#!/usr/bin/env python3
"""Capture real matched inputs, then measure their isolated inference ceiling.

Capture uses simulation Python; measure uses deployment Python with no other
XR-1 instance running. The optional profiler is diagnostic only, after timing.
"""
import argparse
import gzip
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def capture_worker(rank, args):
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      TOKENIZERS_PARALLELISM='false', MUJOCO_GL='osmesa',
                      PYOPENGL_PLATFORM='osmesa', LP_NUM_THREADS='1')
    import numpy as np
    import torch
    from transformers import AutoProcessor
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT/'scripts'))
    from benchmark_robocasa_pipeline import fingerprint
    from eval_robocasa.main import create_env, EvalClient, render_obs, camera_names, rgbs_to_pil_images, center_crop_pil
    torch.set_num_threads(1)
    random.seed(rank)
    np.random.seed(rank)
    torch.manual_seed(rank)
    env=create_env('OpenDrawer', seed=rank)
    try:
        env.reset()
        arm=env.robots[0].composite_controller.part_controllers[env.robots[0].composite_controller.arms[0]]
        base=np.eye(4)
        base[:3,:3],base[:3,3]=arm.origin_ori,arm.origin_pos
        rgb,state,_=render_obs(env,camera_names,base)
        # Reuse identical preparation without opening an inference connection.
        client=EvalClient.__new__(EvalClient)
        client.processor=AutoProcessor.from_pretrained(args.model,trust_remote_code=True,use_fast=False)
        request=client.prepare_request(state,center_crop_pil(rgbs_to_pil_images(rgb),.95),env.get_ep_meta()['lang'])
        expected=np.load(Path(args.tapes)/f'worker-{rank}.npz',allow_pickle=False)['fingerprints'][0]
        if fingerprint(request)!=expected:
            raise RuntimeError(f'Input mismatch for worker {rank}')
        torch.save(request,Path(args.output)/f'input-{rank}.pt')
        return dict(rank=rank,fingerprint=str(expected))
    finally:
        env.close()


def measure(args):
    import torch
    import pynvml as nv
    nv.nvmlInit()
    gpu=nv.nvmlDeviceGetHandleByIndex(0)
    sys.path.insert(0,str(ROOT/'deploy'))
    from batching import XR1BatchPolicy
    torch.set_num_threads(4)
    inputs=[torch.load(p,weights_only=False) for p in sorted(Path(args.inputs).glob('input-*.pt'))]
    if len(inputs)!=32:
        raise RuntimeError('Expected 32 captured observations')
    policy=XR1BatchPolicy(args.model,profile=True)
    for request in inputs:
        policy.validate(request)
    report=[]
    for size in [1,8,16]:
        for _ in range(5):
            policy(inputs[:size])
        durations,stages,hardware=[],[],[]
        for i in range(args.repeats):
            batch=[inputs[(i*size+j)%len(inputs)] for j in range(size)]
            start=time.monotonic()
            policy(batch)
            durations.append(time.monotonic()-start)
            stages.append(dict(policy.last_metrics))
            hardware.append(dict(gpu_util=nv.nvmlDeviceGetUtilizationRates(gpu).gpu,
                                 sm_clock_mhz=nv.nvmlDeviceGetClockInfo(gpu,nv.NVML_CLOCK_SM),
                                 power_watts=nv.nvmlDeviceGetPowerUsage(gpu)/1000))
        row=dict(batch_size=size,seconds=durations,requests_per_s=size*len(durations)/sum(durations),stages=stages,hardware=hardware)
        report.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('seconds','stages','hardware')}),flush=True)
    (Path(args.output)/'cached-inference.json').write_text(json.dumps(report,indent=2))
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                with torch.profiler.record_function('xr1_diagnostic_batch'):
                    policy(inputs[:16])
        trace=Path(args.output)/'cuda-trace.json'
        prof.export_chrome_trace(str(trace))
        with trace.open('rb') as src,gzip.open(str(trace)+'.gz','wb') as dst:
            import shutil
            shutil.copyfileobj(src,dst)
        trace.unlink()
        (Path(args.output)/'operator-times.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=40))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['capture','measure'])
    p.add_argument('--model',required=True)
    p.add_argument('--tapes')
    p.add_argument('--inputs')
    p.add_argument('--output',required=True)
    p.add_argument('--repeats',type=int,default=40)
    p.add_argument('--profile',action='store_true')
    args=p.parse_args()
    Path(args.output).mkdir(parents=True,exist_ok=False)
    if args.mode=='capture':
        with mp.get_context('spawn').Pool(32) as pool:
            rows=pool.starmap(capture_worker,[(rank,args) for rank in range(32)])
        (Path(args.output)/'fingerprints.json').write_text(json.dumps(rows,indent=2))
    else:
        measure(args)


if __name__=='__main__':
    main()
