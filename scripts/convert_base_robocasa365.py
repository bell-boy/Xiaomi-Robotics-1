#!/usr/bin/env python3
"""Package unchanged base inference weights with an explicitly borrowed task adapter.

This performs no training. The resulting artifact is an adapter-based zero-shot
baseline, not a native RoboCasa365 base checkpoint. Training-only choice heads
may be omitted, but every inference tensor must match and load strictly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModel

TRAINING_ONLY_PREFIXES = ('state_projector_choice.', 'action_projector_choice.', 'score_projector_choice.')
# The general release also carries a small action-token embedding and a scalar
# score embedding inside the VLM namespace. The RoboCasa365 architecture uses
# action_projector/score_projector instead and declares neither tensor, so they
# stay unused here. They are named explicitly rather than matched by prefix.
BASE_ONLY_TENSORS = ('vlm.model.action_embed.weight', 'vlm.model.score_embed.weight')


def partition_keys(base_keys, expected_keys):
    """Split base-checkpoint keys into unused, unexpected and missing groups."""
    extra = set(base_keys)-set(expected_keys)
    training_only = sorted(k for k in extra if k.startswith(TRAINING_ONLY_PREFIXES))
    base_only = sorted(k for k in extra if k in BASE_ONLY_TENSORS)
    unexpected = sorted(extra-set(training_only)-set(base_only))
    missing = sorted(set(expected_keys)-set(base_keys))
    return dict(training_only=training_only, base_only=base_only, unexpected=unexpected, missing=missing)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-weights', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for src in args.adapter.iterdir():
        if src.is_file() and src.suffix in ('.py', '.json', '.txt', '.jinja') and src.name != 'model.safetensors.index.json':
            shutil.copy2(src, args.output/src.name)
    cfg = AutoConfig.from_pretrained(args.output, trust_remote_code=True)
    # Preserve the base runtime's +10 offset on generated action positions.
    cfg.inference_action_position_offset = 10
    cfg.zero_shot_adapter = 'RoboCasa365 processor and action statistics; no weight updates'
    with torch.device('meta'):
        model = AutoModel.from_config(cfg, trust_remote_code=True, dtype=torch.bfloat16)
    checkpoint = torch.load(args.base_weights, map_location='cpu', mmap=True, weights_only=False)
    state = checkpoint.get('module', checkpoint)
    state = {k.removeprefix('model.'): v for k, v in state.items()}
    expected = model.state_dict()
    groups = partition_keys(state, expected)
    missing, unexpected = groups['missing'], groups['unexpected']
    mismatched = {k: [list(state[k].shape), list(v.shape)] for k, v in expected.items() if k in state and state[k].shape != v.shape}
    if missing or unexpected or mismatched:
        raise RuntimeError(dict(missing=sorted(missing), unexpected=sorted(unexpected), mismatched=mismatched))
    if groups['base_only'] != sorted(BASE_ONLY_TENSORS):
        raise RuntimeError(f'Base-only tensor set changed: {groups["base_only"]}')
    loaded = {k: state[k] for k in expected}
    model.load_state_dict(loaded, strict=True, assign=True)
    converted_state = model.state_dict()
    assert all(torch.equal(converted_state[k], v) for k, v in loaded.items())
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size='4GB')
    saved_keys = set()
    for shard in args.output.glob('*.safetensors'):
        with safe_open(shard, framework='pt', device='cpu') as f:
            for key in f.keys():
                if not torch.equal(f.get_tensor(key), loaded[key]):
                    raise RuntimeError(f'Saved tensor changed: {key}')
                saved_keys.add(key)
    if saved_keys != set(loaded):
        raise RuntimeError('Saved inference tensor coverage mismatch')
    digest = hashlib.sha256()
    with args.base_weights.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(block)
    report = dict(base_revision='ee21d524b5c52ac961d941e1bc7d6d92836c3d5e',
                  adapter_revision='3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4',
                  source_sha256=digest.hexdigest(), training_steps=0,
                  inference_parameters=sum(v.numel() for v in loaded.values()),
                  all_inference_tensors_unchanged=True,
                  omitted_training_only_keys=groups['training_only'],
                  omitted_base_only_keys=groups['base_only'],
                  action_position_offset=10,
                  caveat='Same RoboCasa365 processor, video history, action mask and mean/std as trained policy. General weights are unchanged. This is an adapter-based zero-shot baseline; robot action conventions may be out of distribution.')
    (args.output/'conversion-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
