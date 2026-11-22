"""Retrospective identifiability of the earlier Costco donor patch; no inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

LAYERS = [0,4,8,12,16,20,23,28,32,36,40,44,47,52,56,60,64,68,71,76,80,84,88,92]
MEASURE = (28,71,92)

def states(path):
    tensor=torch.load(path,map_location="cpu",weights_only=True)["attn_in"]
    if tuple(tensor.shape)!=(24,7168) or not torch.isfinite(tensor).all():
        raise ValueError(f"Invalid matrix in {path.name}")
    return tensor[[LAYERS.index(i) for i in MEASURE]].float().numpy().astype("float64")

def directions(a):
    norm=np.linalg.norm(a,axis=-1,keepdims=True)
    if (norm <= 1e-12).any(): raise ValueError("Empty displacement")
    return a / norm

def read_stage(stage):
    folder=stage/"setup"
    layout=json.loads((folder/"layout.json").read_text())
    entries={}
    for name,cell in layout["cells"].items():
        for u in cell.get("units",[]):
            key=u["capture"]
            if key in entries and entries[key][0]!=name:raise ValueError("Duplicate class")
            entries[key]=(name,u)
    meta=[]; X=[]; Y=[]
    for key,(name,u) in sorted(entries.items()):
        control=states(folder/f'base-{u["receiver"]}.pt')
        reference=states(folder/f'base-{u["donor"]}.pt')-control
        recipient=states(folder/f'arrive-{key}.pt')-control
        X.append(reference);Y.append(recipient)
        meta.append(dict(stage=stage.name,donor=key,condition=name,slot_rows=u["k"],
                         original_text=layout["prompts"][u["donor"]]["text"]))
    return meta,np.array(X),np.array(Y)

def measure(stage):
    meta,X,Y=read_stage(stage); rows=[]
    for j,layer in enumerate(MEASURE):
        sim=directions(Y[:,j])@directions(X[:,j]).T
        for i,m in enumerate(meta):
            by_length=np.array([x["slot_rows"]==m["slot_rows"] for x in meta])
            by_class=np.array([x["condition"]==m["condition"] for x in meta])
            for kind,mask in (("all",np.ones(len(meta),bool)),("length",by_length),
                              ("condition",by_class),("condition-length",by_class&by_length)):
                if mask.sum()<2: continue
                gold=sim[i,i]; neg=mask.copy();neg[i]=False
                tied=(np.abs(sim[i,neg]-gold)<1e-9).sum()
                better=(sim[i,neg]>gold+1e-9).sum()
                rows.append({**m,"layer":layer,"comparison":kind,"n_candidates":int(mask.sum()),
                             "true_cos":float(gold),"hardest_foil":float(sim[i,neg].max()),
                             "rank":float(1+better+tied/2),"unique_top1":int(better==0 and tied==0),
                             "chance":1/mask.sum(),"projection":float(Y[i,j]@X[i,j]/(X[i,j]@X[i,j]))})
    return rows

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('root',type=Path);pa.add_argument('output',type=Path)
    opts=pa.parse_args();opts.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    allrows=[]
    for k in ('k1','k2','k3','k4'):allrows.extend(measure(opts.root/f'cells-{k}'))
    with (opts.output/'donor-ranks.csv').open('w') as f:
        cw=csv.DictWriter(f,fieldnames=list(allrows[0]));cw.writeheader();cw.writerows(allrows)
    by=defaultdict(list)
    for row in allrows:by[row['stage'],row['layer'],row['comparison']].append(row)
    summaries=[]
    for (stage,layer,kind),rs in by.items():
        summaries.append({"stage":stage,"layer":layer,"comparison":kind,"n":len(rs),
                          "top1":sum(x['unique_top1'] for x in rs),
                          "expected_chance_hits":sum(x['chance'] for x in rs),
                          "median_target_cos":float(np.median([x['true_cos'] for x in rs])),
                          "median_foil_cos":float(np.median([x['hardest_foil'] for x in rs])),
                          "median_projection":float(np.median([x['projection'] for x in rs]))})
    (opts.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    for row in summaries:
        if row['layer']==71:print(row)
