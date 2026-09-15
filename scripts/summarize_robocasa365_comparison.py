#!/usr/bin/env python3
"""Validate complete paired evaluations and write Markdown plus video manifests."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_run(path, expected_tasks, trials):
    if not (path/'DONE').exists():
        raise ValueError(f'Run is not complete: {path}')
    summary = json.loads((path/'results/summary.json').read_text())
    if set(summary['tasks']) != set(expected_tasks):
        raise ValueError('Task coverage mismatch')
    videos = []
    for name, task in summary['tasks'].items():
        if task['num_episodes'] != trials or len(task['episodes']) != trials:
            raise ValueError(f'Missing episodes for {name}')
        if sorted(e['episode'] for e in task['episodes']) != list(range(trials)):
            raise ValueError(f'Duplicate/missing episode indices for {name}')
        for ep in task['episodes']:
            status = 'success' if ep['success'] else 'failure'
            video = path/'results'/name/f"episode_{ep['episode']:03d}_seed_{ep['seed']}_{status}.mp4"
            if not video.is_file() or video.stat().st_size == 0:
                raise ValueError(f'Missing video: {video}')
            h = hashlib.sha256()
            with video.open('rb') as f:
                for data in iter(lambda: f.read(4*1024*1024), b''):
                    h.update(data)
            videos.append(dict(task=name, episode=ep['episode'], seed=ep['seed'], success=ep['success'],
                               path=str(video.relative_to(path)), bytes=video.stat().st_size, sha256=h.hexdigest()))
    return summary, videos


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trained', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trials', type=int, default=50)
    p.add_argument('--groups', nargs='+', choices=['atomic_seen', 'composite_seen', 'composite_unseen'], default=['atomic_seen', 'composite_seen', 'composite_unseen'])
    p.add_argument('--drive-url', default='https://drive.google.com/drive/folders/148F8BQ-VHRUCXo6EGJOKRAV5QQTwZIDZ')
    args = p.parse_args()
    groups = json.loads((ROOT/'eval_robocasa365/target_task_groups.json').read_text())
    groups = {name: groups[name] for name in args.groups}
    tasks = [task for group in groups.values() for task in group]
    trained, trained_videos = load_run(args.trained, tasks, args.trials)
    base, base_videos = load_run(args.base, tasks, args.trials)
    for name in tasks:
        a = sorted(trained['tasks'][name]['episodes'], key=lambda r:r['episode'])
        b = sorted(base['tasks'][name]['episodes'], key=lambda r:r['episode'])
        if [(e['episode'],e['seed']) for e in a] != [(e['episode'],e['seed']) for e in b]:
            raise ValueError(f'Unpaired seeds for {name}')
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for group, names in groups.items():
        for name in names:
            a, b = trained['tasks'][name], base['tasks'][name]
            rows.append(dict(group=group, task=name, episodes_per_policy=args.trials,
                             trained_successes=a['successes'], base_successes=b['successes'],
                             trained_success_rate=a['success_rate'], base_success_rate=b['success_rate'],
                             difference_pp=100*(a['success_rate']-b['success_rate'])))
    with (args.output/'tasks.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report=dict(groups=groups, tasks=rows, trained_summary=trained, base_summary=base,
                trained_videos=trained_videos, base_videos=base_videos)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2))
    lines=['# XR-1 RoboCasa365 policy comparison','',
           'Complete paired evaluation of the released RoboCasa365-trained policy and unchanged general XR-1-5B inference weights.','',
           '**Base-policy interpretation:** the base run uses the RoboCasa365 processor, observation/action interface and action normalization statistics without any weight updates. It is an adapter-based zero-shot test, not a published native base-policy benchmark. Out-of-distribution robot conventions can affect its score.','',
           '## Results','',
           '| Task group | Tasks | Episodes per policy | RoboCasa365-trained | Base, no fine-tuning | Difference |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label,names in list(groups.items())+[('Overall',tasks)]:
        n=len(names)*args.trials
        a=sum(trained['tasks'][t]['successes'] for t in names)
        b=sum(base['tasks'][t]['successes'] for t in names)
        lines.append(f'| {label} | {len(names)} | {n} | {a}/{n} ({100*a/n:.2f}%) | {b}/{n} ({100*b/n:.2f}%) | {100*(a-b)/n:+.2f} pp |')
    lines+=['','Rates average equally over tasks because every task has the same episode count.','',
            '## Protocol','',
            f'- {args.trials} episodes per task, target split, RoboCasa v1.0.1 official horizons; matching initial seeds and per-request action-noise seeds.',
            '- Four RTX 5880 Ada 48 GB GPUs, one policy instance per GPU, 20 ms batching; actual run settings are saved in each run-config.json.',
            '- Independent OSMesa simulators, observation history 4 at interval 2, 16 actions per query, crop ratio 0.95.',
            '- All episodes recorded as three-camera MP4s, stride 2 at 20 FPS. Videos therefore play faster than real time when simulation control frequency is 20 Hz.',
            '- Checkpoints: RoboCasa365 revision 3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4; base revision ee21d524b5c52ac961d941e1bc7d6d92836c3d5e.',
            '- The base inference tensor equality check and excluded training-only heads are documented in conversion-report.json. No fine-tuning was performed.',
            '- Comparison with published scores requires matching renderer, task split, horizon version and stochastic sampling protocol; this run is a paired local comparison.','',
            '## Videos and data','',f'[Complete videos in Google Drive]({args.drive_url}).',
            'The JSON comparison includes a checksum and byte count for every video. Per-task results are in tasks.csv.','',
            '## Per-task results','',
            '| Group | Task | Trained successes | Base successes | Difference |',
            '| --- | --- | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['group']} | {r['task']} | {r['trained_successes']}/{args.trials} | {r['base_successes']}/{args.trials} | {r['difference_pp']:+.2f} pp |")
    (args.output/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print(f'Validated {len(trained_videos)+len(base_videos)} videos and paired episodes.')


if __name__ == '__main__':
    main()
