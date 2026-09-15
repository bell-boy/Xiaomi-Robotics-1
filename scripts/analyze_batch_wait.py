#!/usr/bin/env python3
"""Summarize the common steady-state interval with all clients active."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values)-1)*q/100
    lo = int(index)
    hi = min(lo+1, len(values)-1)
    return values[lo] + (values[hi]-values[lo])*(index-lo)


def stats(values):
    values = list(values)
    if not values:
        return dict(count=0, mean=None, p50=None, p95=None, p99=None, max=None)
    return dict(count=len(values), mean=statistics.mean(values), p50=percentile(values, 50),
                p95=percentile(values, 95), p99=percentile(values, 99), max=max(values))


def jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def overlap(start, end, lo, hi):
    return max(0, min(end,hi)-max(start,lo))


def summarize_trial(path, cpu_quota):
    config = json.loads((path/'clients/config.json').read_text())
    completion = json.loads((path/'clients/completion.json').read_text())
    if len(completion) != config['workers'] or not all(row.get('workload_verified') for row in completion):
        raise RuntimeError(f'Incomplete or mismatched workload: {path}')
    workers = [json.loads((path/f'clients/worker-{rank}.json').read_text()) for rank in range(config['workers'])]
    if any(len(rows) != config['warmup']+config['calls'] for rows in workers):
        raise RuntimeError('Missing worker calls')
    # Independent warmup and completion. Trim to the intersection where every
    # worker has completed warmup and no worker has yet stopped issuing work.
    lo = max(rows[config['warmup']]['cycle_start'] for rows in workers)
    hi = min(rows[-1]['cycle_end'] for rows in workers)
    if hi <= lo:
        raise RuntimeError('No common steady-state interval')
    duration = hi-lo
    all_rows = [row for worker in workers for row in worker]
    full_cycles = [row for row in all_rows if lo <= row['cycle_start'] and row['cycle_end'] <= hi]
    requests = [row for row in all_rows if lo <= row['request_start'] and row['response_end'] <= hi]
    step_times = [t for row in all_rows for t in row['step_ends'] if lo <= t < hi]
    completed_requests = sum(lo <= row['response_end'] < hi for row in all_rows)
    full_start = min(rows[config['warmup']]['cycle_start'] for rows in workers)
    full_end = max(rows[-1]['cycle_end'] for rows in workers)
    metrics = jsonl(path/'server.jsonl')
    errors = [r for r in metrics if r['kind']=='batch_error']
    if errors:
        raise RuntimeError(f'Inference errors: {errors[:2]}')
    all_batches = sorted([r for r in metrics if r['kind']=='batch'], key=lambda r:r['start'])
    batches = [r for r in all_batches if lo <= r['start'] and r['end'] <= hi]
    server_requests = [r for r in metrics if r['kind']=='request' and lo <= r['received_start'] and r['sent_at'] <= hi]
    queue = [r['depth'] for r in metrics if r['kind']=='queue_sample' and lo <= r['time'] < hi]
    gpu_all = jsonl(path/'gpu.jsonl')
    gpu = [r for r in gpu_all if lo <= r['time'] < hi]
    if len(gpu) < 2:
        raise RuntimeError('Missing GPU telemetry')
    busy_s = sum(overlap(r['start'],r['end'],lo,hi) for r in all_batches)
    collect_s = sum(overlap(r['collect_start'],r['start'],lo,hi) for r in all_batches)
    gaps, cursor = [], lo
    for r in all_batches:
        if r['end'] < lo or r['start'] > hi:
            continue
        if r['start'] > cursor:
            gaps.append((min(r['start'],hi)-cursor)*1000)
        cursor=max(cursor,min(r['end'],hi))
    if cursor < hi:
        gaps.append((hi-cursor)*1000)
    distribution=Counter(r['size'] for r in batches)
    batch_requests=sum(r['size'] for r in batches)
    batch_seconds=sum(r['end']-r['start'] for r in batches)
    by_size={}
    for size in sorted(distribution):
        selected=[r for r in batches if r['size']==size]
        by_size[str(size)] = dict(count=len(selected), inference_ms=stats((r['end']-r['start'])*1000 for r in selected))
    # Hardware telemetry is a sampled indicator, not a kernel-level GPU trace.
    low_gpu_runs=[]
    low_start=None
    for r in gpu:
        if r['gpu_util'] < 5:
            if low_start is None:
                low_start=r['time']
        elif low_start is not None:
            low_gpu_runs.append((r['time']-low_start)*1000)
            low_start=None
    if low_start is not None:
        low_gpu_runs.append((gpu[-1]['time']-low_start)*1000)
    blocks=[]
    left=lo
    while left+30 <= hi:
        right=left+30
        blocks.append(dict(start=left-lo, duration=30,
                           steps_per_s=sum(left <= t < right for t in step_times)/30))
        left=right
    sample_duration=gpu[-1]['time']-gpu[0]['time']
    cpu_cores=(gpu[-1]['cpu_ns']-gpu[0]['cpu_ns'])/1e9/sample_duration if gpu[-1]['cpu_ns'] is not None else None
    throttle=None
    if 'cpu_throttle' in gpu[0] and 'cpu_throttle' in gpu[-1]:
        start,end=gpu[0]['cpu_throttle'],gpu[-1]['cpu_throttle']
        periods=int(end['nr_periods'])-int(start['nr_periods'])
        throttle=(int(end['nr_throttled'])-int(start['nr_throttled']))/max(1,periods)*100
    wait=int(path.name.rsplit('-',1)[1])
    return dict(trial=path.name, wait_ms=wait, workers=config['workers'], max_batch=16,
        mode=config['mode'], warmup_calls_per_worker=config['warmup'], measured_calls_per_worker=config['calls'],
        workload_verified=True, steady_start=lo, steady_end=hi, steady_seconds=duration,
        completed_steps=len(step_times), completed_requests=completed_requests,
        steps_per_s=len(step_times)/duration, requests_per_s=completed_requests/duration,
        fixed_work_steps_per_s=config['workers']*config['calls']*10/(full_end-full_start),
        fixed_work_duration_s=full_end-full_start,
        batch_size_distribution=dict(sorted(distribution.items())),
        batch_size=stats(r['size'] for r in batches),
        full_batch_percent=distribution[16]/max(1,len(batches))*100,
        request_weighted_batch_size=sum(r['size']**2 for r in batches)/max(1,batch_requests),
        inference_ms=stats((r['end']-r['start'])*1000 for r in batches),
        inference_busy_requests_per_s=batch_requests/batch_seconds,
        inference_busy_percent=busy_s/duration*100,
        collection_percent=collect_s/duration*100,
        other_dispatch_gap_percent=max(0,duration-busy_s-collect_s)/duration*100,
        dispatch_gap_ms=stats(gaps), collection_ms=stats(r['collection_ms'] for r in batches),
        queue_depth=stats(queue), queue_zero_percent=sum(x==0 for x in queue)/max(1,len(queue))*100,
        queue_wait_ms=stats(t for r in batches for t in r['queue_wait_ms']),
        request_latency_ms=stats((r['response_end']-r['request_start'])*1000 for r in requests),
        cycle_ms=stats((r['cycle_end']-r['cycle_start'])*1000 for r in full_cycles),
        client_stages={key:stats(r[key] for r in full_cycles) for key in
            ['render_ms','preprocess_ms','fingerprint_ms','serialize_ms','send_ms','response_wait_ms',
             'unpickle_ms','decode_ms','physics_ms','cpu_ms']},
        server_transfer={
            'receive_ms':stats((r['received_end']-r['received_start'])*1000 for r in server_requests),
            'unpickle_ms':stats((r['decoded_at']-r['received_end'])*1000 for r in server_requests),
            'serialize_ms':stats((r['serialized_at']-r['result_at'])*1000 for r in server_requests),
            'send_ms':stats((r['sent_at']-r['serialized_at'])*1000 for r in server_requests),
            'input_bytes':stats(r['input_bytes'] for r in server_requests)},
        gpu_stages={key:stats(r['stages'][key] for r in batches if key in r['stages']) for key in
            ['collate_ms','h2d_wall_ms','h2d_stream_ms','vlm_stream_ms','action_head_stream_ms','d2h_stream_ms']},
        gpu_util=stats(r['gpu_util'] for r in gpu), gpu_low_util_percent=sum(r['gpu_util']<5 for r in gpu)/len(gpu)*100,
        gpu_low_util_gap_ms=stats(low_gpu_runs), gpu_memory_peak_gib=max(r['gpu_memory_bytes'] for r in gpu)/2**30,
        ram_peak_gib=max(r['ram_bytes'] for r in gpu_all)/2**30,
        cpu_cores_used=cpu_cores, cpu_quota_util_percent=None if cpu_cores is None else cpu_cores/cpu_quota*100,
        cpu_throttled_periods_percent=throttle,
        gpu_power_watts=stats(r['power_watts'] for r in gpu),
        gpu_sm_clock_mhz=stats(r['sm_clock_mhz'] for r in gpu),
        gpu_temperature_c=stats(r['temperature_c'] for r in gpu),
        by_batch_size=by_size, blocks_30s=blocks)


def summarize(root):
    metadata=json.loads((root/'metadata.json').read_text())
    quota=metadata['cpu_quota']/metadata['cpu_period']
    trials=[summarize_trial(p,quota) for p in sorted(root.glob('*-wait-*')) if (p/'DONE').exists()]
    grouped=defaultdict(list)
    for trial in trials:
        grouped[trial['wait_ms']].append(trial)
    pooled=[]
    for wait,rows in sorted(grouped.items()):
        seconds=sum(r['steady_seconds'] for r in rows)
        pooled.append(dict(wait_ms=wait, runs=len(rows), steady_seconds=seconds,
                           steps_per_s=sum(r['completed_steps'] for r in rows)/seconds,
                           requests_per_s=sum(r['completed_requests'] for r in rows)/seconds,
                           run_steps_per_s=[r['steps_per_s'] for r in rows],
                           fixed_work_steps_per_s=sum(r['workers']*r['measured_calls_per_worker']*10 for r in rows)/sum(r['fixed_work_duration_s'] for r in rows)))
    return dict(method='Common steady-state interval; all 32 clients independently active after individual warmup; matched action tape with exact input hashes',
                trials=trials, pooled=pooled,
                best_wait_ms=max(pooled,key=lambda r:r['steps_per_s'])['wait_ms'] if pooled else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    report=summarize(args.directory)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))
    fields=['trial','wait_ms','steady_seconds','steps_per_s','requests_per_s','fixed_work_steps_per_s',
            'batch_mean','batch_full_percent','gpu_util_mean','gpu_low_util_percent',
            'inference_busy_percent','collection_percent','queue_mean','queue_p95',
            'latency_p50_ms','latency_p95_ms','cpu_quota_util_percent']
    with (args.output/'trials.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for r in report['trials']:
            row={k:r[k] for k in fields if k in r}
            row.update(batch_mean=r['batch_size']['mean'],batch_full_percent=r['full_batch_percent'],
                       gpu_util_mean=r['gpu_util']['mean'],queue_mean=r['queue_depth']['mean'],
                       queue_p95=r['queue_depth']['p95'],latency_p50_ms=r['request_latency_ms']['p50'],
                       latency_p95_ms=r['request_latency_ms']['p95'])
            writer.writerow(row)
    print(json.dumps(report['pooled'],indent=2))


if __name__=='__main__':
    main()
