#!/usr/bin/env python3
"""Sequential, repeated, matched-workload batching-window experiment on one host.
Run with the deployment Python. Stop the normal server before starting this.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def read_number(path):
    try:
        return int(Path(path).read_text())
    except (OSError, ValueError):
        return None


def monitor(path, stop):
    import pynvml as nv
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByIndex(0)
    with path.open('w', buffering=1) as f:
        while not stop.is_set():
            memory = nv.nvmlDeviceGetMemoryInfo(handle)
            util = nv.nvmlDeviceGetUtilizationRates(handle)
            row = dict(time=time.monotonic(), gpu_util=util.gpu, memory_util=util.memory,
                       gpu_memory_bytes=memory.used, power_watts=nv.nvmlDeviceGetPowerUsage(handle)/1000,
                       temperature_c=nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU),
                       sm_clock_mhz=nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM),
                       cpu_ns=read_number('/sys/fs/cgroup/cpuacct/cpuacct.usage'),
                       ram_bytes=read_number('/sys/fs/cgroup/memory/memory.usage_in_bytes'))
            try:
                row['cpu_throttle'] = dict(line.split() for line in Path('/sys/fs/cgroup/cpu/cpu.stat').read_text().splitlines())
            except OSError:
                pass
            f.write(json.dumps(row, separators=(',', ':'))+'\n')
            stop.wait(.2)
    nv.nvmlShutdown()


def stop_process(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--sim-python', default='/opt/conda/envs/robocasa/bin/python')
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--calls', type=int, default=160)
    p.add_argument('--windows', nargs='+', type=int, default=[20, 100, 50, 50, 100, 20])
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Fail early if telemetry is unavailable, rather than silently losing it.
    import pynvml
    pynvml.nvmlInit()
    pynvml.nvmlShutdown()
    metadata = dict(args=vars(args), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    gpu=subprocess.check_output(['nvidia-smi','-q'],text=True),
                    cpu_quota=read_number('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'),
                    cpu_period=read_number('/sys/fs/cgroup/cpu/cpu.cfs_period_us'),
                    ram_limit=read_number('/sys/fs/cgroup/memory/memory.limit_in_bytes'))
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2))
    server = clients = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for index, wait in enumerate(args.windows):
            trial = output / f'{index+1:02d}-wait-{wait}'
            trial.mkdir()
            print(f'Starting {trial.name}', flush=True)
            env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
            with (trial/'server.log').open('w') as log:
                server = subprocess.Popen([sys.executable, '-u', 'deploy/server.py',
                    '--model', args.model, '--host', '127.0.0.1', '--port', '10086',
                    '--max-batch-size', '16', '--batch-wait-ms', str(wait),
                    '--metrics-jsonl', str(trial/'server.jsonl')], cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic()+180
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Server failed; see {trial / "server.log"}')
                try:
                    with socket.create_connection(('127.0.0.1',10086), timeout=1):
                        break
                except OSError:
                    if time.monotonic()>deadline:
                        raise TimeoutError('Server did not become ready')
                    time.sleep(.5)
            stop = threading.Event()
            sampler = threading.Thread(target=monitor, args=(trial/'gpu.jsonl', stop))
            sampler.start()
            try:
                with (trial/'clients.log').open('w') as log:
                    clients = subprocess.Popen([args.sim_python, '-u', 'scripts/benchmark_robocasa_pipeline.py',
                        '--model', args.model, '--workers', str(args.workers), '--warmup', str(args.warmup),
                        '--calls', str(args.calls), '--mode', 'record' if index == 0 else 'replay',
                        '--tape-dir', str(output/'tapes'), '--output', str(trial/'clients')], cwd=ROOT,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                code = clients.wait()
                if code:
                    raise RuntimeError(f'Clients failed ({code}); see {trial / "clients.log"}')
            finally:
                stop.set()
                sampler.join(timeout=5)
                stop_process(clients)
                stop_process(server)
            (trial/'DONE').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())
            print(f'Completed {trial.name}', flush=True)
        (output/'DONE').write_text('All trials completed\n')
    finally:
        stop_process(clients)
        stop_process(server)


if __name__ == '__main__':
    main()
