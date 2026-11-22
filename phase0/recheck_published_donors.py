"""Reproduce all archival top-1 frequencies solely from the release package."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def check(audit):
    description=json.loads((audit/'native-rows/native-rows.json').read_text())
    summary=json.loads((audit/'summary.json').read_text())
    verified=0
    for stage,info in description['stages'].items():
        p=audit/'native-rows'/info['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
        with np.load(p,allow_pickle=False) as z:
            dots=z['dots'].astype('float64')
            x=z['donor'].astype('float64')-dots
            y=z['patched'].astype('float64')-dots
        x/=np.linalg.norm(x,axis=-1,keepdims=True);y/=np.linalg.norm(y,axis=-1,keepdims=True)
        labels=np.array([r['condition'] for r in info['rows']]);length=np.array([r['slot_rows'] for r in info['rows']])
        for j,layer in enumerate(description['layers']):
            cos=np.einsum('ih,jh->ij',y[:,j,:],x[:,j,:],optimize=False)
            for restriction in ('all','length','condition','condition-length'):
                hits=0;total=0
                for i,row in enumerate(cos):
                    mask=np.ones(len(row),dtype=bool)
                    if 'length' in restriction:mask &= length==length[i]
                    if 'condition' in restriction:mask &= labels==labels[i]
                    mask[i]=False
                    if not mask.any(): continue
                    total+=1;hits+=int(row[i]>np.max(row[mask])+1e-9)
                a=next(s for s in summary if s['stage']=='cells-'+stage and s['layer']==layer and s['comparison']==restriction)
                assert (hits,total)==(a['top1'],a['n']),(stage,layer,restriction)
                verified+=1
    print(f"Reproduced all {verified} layer-by-stage-by-candidate top-1 findings from release tensors.")
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('audit',type=Path);check(ap.parse_args().audit)
