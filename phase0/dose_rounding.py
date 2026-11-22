"""BF16 dose rehearsal on pre-existing native states from another stimulus set."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def summary(values):
    a = np.asarray(values, dtype=float)
    return {k: float(np.quantile(a, q)) for k, q in (("min",0),("q05",.05),("median",.5),("q95",.95),("max",1))}


def run(src, out):
    torch.set_num_threads(1)
    src = Path(src)
    d = src / "smell-s1"
    meta = json.loads((d/"activations-s1.json").read_text())
    prompts = {v["key"]:v for v in json.loads((d/"prompts.json").read_text())["items"]}
    items = [prompts[k] for k in meta["keys"]]
    arr = np.memmap(d/"activations-s1.f16",mode="r",dtype=np.float16,shape=tuple(meta["shape"]))
    candidates = sorted({v['object'] for v in items if v['condition']=='smell'} &
                        {v['object'] for v in items if v['condition']=='see'})
    pos = meta["positions"].index("end_of_msg")
    site = meta["sites"].index("attn_in")
    recipients=sorted((src/"cells-k1/setup").glob("base-costco-dots*.pt"))
    rows = []
    for L in (23,47,71):
        j=meta['layers'].index(L)
        for obj in candidates:
            def centroid(kind):
                idx=[i for i,v in enumerate(items) if v['object']==obj and v['condition']==kind]
                return arr[site,j,idx,pos,:].astype(np.float64).mean(0)
            d0=centroid('smell')-centroid('see')
            seed=int(hashlib.sha256(f'{obj}:{L}'.encode()).hexdigest()[:8],16)
            wrong=np.random.default_rng(seed).permutation(d0)
            for pf in recipients:
                h=torch.load(pf,map_location='cpu',weights_only=True)['attn_in'][j].to(torch.bfloat16)
                norm=float(h.float().norm())
                for kind,delta in (('archived_smell_minus_see',d0),('permuted',wrong)):
                    intended=0.05*delta
                    clipped=intended*min(1,.01*norm/np.linalg.norm(intended))
                    v=torch.from_numpy(clipped.astype('float32'))
                    applied=(h+v.to(h.dtype)).float()-h.float()
                    relative=float(applied.norm()/norm)
                    rows.append(dict(layer=L,object=obj,recipient=pf.stem,arm=kind,
                                     requested_fraction=float(v.norm()/norm),realized_fraction=relative,
                                     ratio=float(applied.norm()/v.norm()),
                                     direction_cos=float(torch.nn.functional.cosine_similarity(applied,v,dim=0)),
                                     changed_fraction=float((applied!=0).float().mean()),exceeds_limit=relative>.01))
    out.mkdir(parents=True,exist_ok=True)
    with (out/'rows.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    report={"scope":"No model forward. Archived independent recipient states and smell-minus-sight vectors; these are not the prospective learned writers.",
            "reference_position":"end_of_msg","recipient_position":"turn_start",
            "objects":candidates,"recipient_count":len(recipients),"site":"attn_in","rows":len(rows),
            "exceeded":sum(r['exceeds_limit'] for r in rows)}
    for key in ('realized_fraction','requested_fraction','ratio','direction_cos','changed_fraction'):report[key]=summary([r[key] for r in rows])
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.source,a.output)
