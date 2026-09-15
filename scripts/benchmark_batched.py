#!/usr/bin/env python3
"""GPU correctness and throughput check using three-camera synthetic inputs.
This measures inference, not RoboCasa simulator throughput or success rates.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy'))
from batching import XR1BatchPolicy


def make_request(processor, index=0):
    rng = np.random.default_rng(index)
    images = [Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)) for _ in range(3)]
    instruction = ['Open the drawer.', 'Pick up the mug and place it on the kitchen counter.'][index % 2]
    messages = [
        {'role': 'user', 'content': [
            {'type': 'text', 'text': 'The following observations are captured from multiple views.\n# Base View\n'},
            {'type': 'image', 'image': images[0]}, {'type': 'image', 'image': images[1]},
            {'type': 'text', 'text': '\n# Left-Wrist View\n'}, {'type': 'image', 'image': images[2]},
            {'type': 'text', 'text': f'\nGenerate robot actions for the task:\n{instruction} /no_cot'}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': '<cot></cot>'}]}]
    data = dict(processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors='pt',
        state=rng.normal(0, .1, (1, 1, 60)).astype(np.float32), robot_type='robocasa_mg'))
    data.update(task_id='robocasa_mg', seed=42 + index)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 8, 16, 32])
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(4)
    policy = XR1BatchPolicy(args.model)
    requests = [make_request(policy.processor, i) for i in range(max(args.batch_sizes))]
    for request in requests:
        policy.validate(request)
    # Compare the published forward to the adapter before measuring batches.
    reference_data = {k: v.to(device=policy.model.device,
                             dtype=policy.model.dtype if v.is_floating_point() else v.dtype)
                      if isinstance(v, torch.Tensor) else v for k, v in requests[0].items()}
    with torch.inference_mode():
        reference = policy.model(**reference_data).actions.cpu()
    serial = [policy([r])[0] for r in requests[:2]]
    torch.testing.assert_close(serial[0], reference, atol=.01, rtol=.01)
    pair = policy(requests[:2])
    differences = []
    for single, batched in zip(serial, pair):
        differences.append(float((single.float() - batched.float()).abs().max()))
        torch.testing.assert_close(single, batched, atol=.05, rtol=.05)
    report = {'device': torch.cuda.get_device_name(),
              'input_shapes': {k: list(v.shape) for k, v in requests[0].items() if isinstance(v, torch.Tensor)},
              'singleton_reference_max_abs': float((serial[0].float() - reference.float()).abs().max()),
              'mixed_length_pair_max_abs': differences, 'benchmarks': []}
    for bs in args.batch_sizes:
        policy(requests[:bs])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        durations = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            outputs = policy(requests[:bs])
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)
            assert len(outputs) == bs and all(torch.isfinite(x).all() for x in outputs)
        mean = sum(durations) / len(durations)
        row = {'batch_size': bs, 'mean_seconds': mean, 'requests_per_second': bs / mean,
               'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
        report['benchmarks'].append(row)
        print(json.dumps(row), flush=True)
        Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
