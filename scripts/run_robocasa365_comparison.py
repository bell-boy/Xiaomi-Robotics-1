#!/usr/bin/env python3
"""Run the paired RoboCasa365 comparison with one episodes-per-task knob.

Wraps the trained-policy evaluation, the base-weight conversion, the base-policy
evaluation and the comparison summary so that every stage uses the same episode
count. Change ``--episodes-per-task`` and nothing else.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TRAINED_RUN = 'robocasa365-trained'
BASE_RUN = 'base-5b-no-finetuning'
TASK_SETS = ['target50', 'atomic_seen', 'composite_seen', 'composite_unseen']


def plan(args):
    """Return the stages to execute, as (name, description, command) triples."""
    root = args.evaluations_root/args.run_tag
    trained_dir = root/TRAINED_RUN
    base_dir = root/BASE_RUN
    adapter = args.base_adapter or args.trained_model
    stages = []

    def launcher(model, output):
        command = [sys.executable, '-u', str(ROOT/'scripts/launch_robocasa365_batched.py'),
                   '--model', str(model), '--output', str(output), '--sim-python', args.sim_python,
                   '--gpus', *[str(gpu) for gpu in args.gpus],
                   '--workers-per-gpu', str(args.workers_per_gpu),
                   '--max-batch-size', str(args.max_batch_size),
                   '--batch-wait-ms', str(args.batch_wait_ms),
                   '--task-set', args.task_set, '--split', args.split,
                   '--num-trials', str(args.episodes_per_task)]
        for task in args.task_name or []:
            command += ['--task-name', task]
        if args.horizon is not None:
            command += ['--horizon', str(args.horizon)]
        return command

    if 'trained' in args.policies:
        stages.append(('trained', f'{args.episodes_per_task} episodes/task, {args.task_set}',
                       launcher(args.trained_model, trained_dir)))
    if 'base' in args.policies:
        converted = args.base_converted_model or (root/'base-converted-model')
        if not args.skip_conversion:
            stages.append(('convert', f'package base weights with the {adapter.name} adapter',
                           [sys.executable, '-u', str(ROOT/'scripts/convert_base_robocasa365.py'),
                            '--base-weights', str(args.base_weights), '--adapter', str(adapter),
                            '--output', str(converted)]))
        stages.append(('base', f'{args.episodes_per_task} episodes/task, {args.task_set}',
                       launcher(converted, base_dir)))
    if {'trained', 'base'} <= set(args.policies):
        stages.append(('summary', 'paired comparison, videos and checksums',
                       [sys.executable, '-u', str(ROOT/'scripts/summarize_robocasa365_comparison.py'),
                        '--trained', str(trained_dir), '--base', str(base_dir),
                        '--output', str(root/'comparison'), '--groups', args.task_set,
                        '--trials', str(args.episodes_per_task)]))
    return root, stages


def reset(output, root):
    """Remove an existing run directory after checking it is inside the run root."""
    if not output.exists():
        return
    if root not in output.parents:
        raise SystemExit(f'Refusing to delete {output}: outside {root}')
    shutil.rmtree(output)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes-per-task', '--num-trials', dest='episodes_per_task', type=int, default=10,
                   help='Episodes per task for both policies. The single knob for evaluation size.')
    p.add_argument('--trained-model', type=Path, default=Path('/workspace/checkpoints/Xiaomi-Robotics-1-RoboCasa365'))
    p.add_argument('--base-weights', type=Path, default=Path('/workspace/checkpoints/Xiaomi-Robotics-1-5B/model_states.pt'))
    p.add_argument('--base-converted-model', type=Path, default=None,
                   help='Where to write the base weights plus adapter. Defaults to <run root>/base-converted-model.')
    p.add_argument('--base-adapter', type=Path, default=None, help='Defaults to --trained-model.')
    p.add_argument('--evaluations-root', type=Path, default=Path('/workspace/evaluations'))
    p.add_argument('--run-tag', default=None, help='Defaults to <task set>-<episodes per task>.')
    p.add_argument('--sim-python', default='/workspace/robocasa365-eval-venv/bin/python')
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    p.add_argument('--workers-per-gpu', type=int, default=8)
    p.add_argument('--max-batch-size', type=int, default=16)
    p.add_argument('--batch-wait-ms', type=float, default=20)
    p.add_argument('--split', choices=['pretrain', 'target'], default='target')
    p.add_argument('--task-set', choices=TASK_SETS, default='atomic_seen')
    p.add_argument('--task-name', action='append', help='Optional task subset; repeatable.')
    p.add_argument('--horizon', type=int, help='Optional horizon override, for smoke tests only.')
    p.add_argument('--policies', nargs='+', choices=['trained', 'base'], default=['trained', 'base'])
    p.add_argument('--skip-conversion', action='store_true', help='Reuse an existing converted base model.')
    p.add_argument('--force', action='store_true', help='Replace an existing run directory for the same tag.')
    p.add_argument('--dry-run', action='store_true', help='Print the stages without running them.')
    return p


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def main():
    p = build_parser()
    args = p.parse_args()

    if args.episodes_per_task < 1:
        p.error('--episodes-per-task must be at least 1')
    if len(set(args.gpus)) != len(args.gpus):
        p.error('GPU indices must be distinct')
    if args.run_tag is None:
        args.run_tag = f'{args.task_set}-{args.episodes_per_task}'
    args.evaluations_root = args.evaluations_root.resolve()
    root, stages = plan(args)
    print(f'Run root: {root}')
    print(f'Policies: {" then ".join(args.policies)}, {args.episodes_per_task} episodes per task, '
          f'{args.task_set} tasks on the {args.split} split, {len(args.gpus)} GPUs')
    for name, description, command in stages:
        print(f'  [{name}] {description}')
    if args.dry_run:
        for name, _, command in stages:
            print(f'\n# {name}\n' + ' '.join(command))
        return

    if root.exists():
        if not args.force:
            raise SystemExit(f'{root} already exists; move it, pick another --run-tag, or pass --force')
        for stage_name in ['trained', 'base']:
            if stage_name in args.policies:
                reset(root/(TRAINED_RUN if stage_name == 'trained' else BASE_RUN), root)
    root.mkdir(parents=True, exist_ok=True)
    (root/'run-plan.json').write_text(json.dumps(
        dict(episodes_per_task=args.episodes_per_task, task_set=args.task_set, split=args.split,
             policies=args.policies, stages=[dict(name=n, description=d, command=c) for n, d, c in stages]),
        indent=2))
    for name, description, command in stages:
        print(f'\n=== {name}: {description} ===', flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    (root/'DONE').write_text(f'{args.episodes_per_task} episodes per task, policies: {", ".join(args.policies)}\n')
    print(f'\nComplete: {root}')


if __name__ == '__main__':
    main()
