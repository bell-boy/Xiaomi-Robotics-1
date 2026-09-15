#!/usr/bin/env python3
"""Check the video-history sampler against the released singleton forward."""
import argparse
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deploy'))
from batching import XR1BatchPolicy, collate

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',required=True)
p.add_argument('--requests',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
policy=XR1BatchPolicy(a.model)
rows=[torch.load(f,weights_only=False) for f in sorted(a.requests.glob('*.pt'))[:2]]
if len(rows)<2:
    raise RuntimeError('Need two real requests')
for r in rows: policy.validate(r)
with torch.inference_mode():
    singles=[policy([r])[0] for r in rows]
    r=rows[0]
    data={k:v.to(device='cuda',dtype=policy.model.dtype if v.is_floating_point() else v.dtype)
          for k,v in collate([r],policy.pad_token_id).items()}
    torch.manual_seed(r['seed'])
    reference=policy.model(**data).actions.cpu()
    singleton_error=float((singles[0]-reference).abs().max())
    if not torch.allclose(singles[0],reference,atol=.01,rtol=.01):
        raise RuntimeError(f'Singleton/reference mismatch: {singleton_error}')
    batch=policy(rows)
    mixed_errors=[float((s-b).abs().max()) for s,b in zip(singles,batch)]
    if not all(torch.allclose(s,b,atol=.1,rtol=.05) for s,b in zip(singles,batch)):
        raise RuntimeError(f'Batch/singleton mismatch: {mixed_errors}')
report=dict(singleton_reference_max_abs=singleton_error,mixed_batch_max_abs=mixed_errors,
            request_shapes=[{k:list(v.shape) for k,v in r.items() if isinstance(v,torch.Tensor)} for r in rows],
            peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
