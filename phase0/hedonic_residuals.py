"""Training-only residualization of an external human quality criterion."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

def fit(x,y):
    X=np.column_stack([np.ones(len(x)),x])
    return np.linalg.lstsq(X,y,rcond=None)[0]

def predict(x,b):
    X=np.column_stack([np.ones(len(x)),x])
    return np.einsum('ij,jk->ik',X,b,optimize=False)

def main(st,out):
    roster=json.loads((st/'roster.json').read_text())
    landmarks=json.loads((st/'landmarks.json').read_text());schema=json.loads((st/'manifest.json').read_text())
    dims=sorted(schema['quality_sd_in_fitting'])
    idx={r['cid']:i for i,r in enumerate(roster)}
    train=np.array([r['split']=='fit' for r in roster])
    x=np.array([[r[k] for k in ('intensity','pleasantness','familiarity')] for r in roster],float)
    x=(x-x[train].mean(0))/x[train].std(0)
    y=np.array([[r['quality'][k] for k in dims] for r in roster],float)
    y=y/np.array([schema['quality_sd_in_fitting'][k] for k in dims])
    bt=fit(x[train],y[train]);resid=y-predict(x,bt)
    comp=[]
    with (st/'human_choices.csv').open() as f:
        h={(r['cid'],r['pair_id']):float(r['human_margin']) for r in csv.DictReader(f)}
    for r in roster:
        if r['split']!='test':continue
        t=idx[r['cid']]
        for l in landmarks:
            if l['role']!='test':continue
            a,b=idx[l['left']],idx[l['right']]
            def margin(z):return np.linalg.norm(z[t]-z[b])-np.linalg.norm(z[t]-z[a])
            raw=margin(y);adjust=margin(resid)
            if abs(raw-h[r['cid'],l['pair_id']])>1e-8:raise ValueError("The raw external benchmark does not reproduce")
            comp.append({'cid':r['cid'],'reference_pair':l['pair_id'],'published_margin':raw,
                         'nuisance_residual_margin':adjust,'same_sign':int(np.sign(raw)==np.sign(adjust))})
    folds=defaultdict(list)
    for i,r in enumerate(roster):
        if train[i]:folds[r['family']].append(i)
    predicted=[];actual=[];null=[]
    for family,held in folds.items():
        other=train.copy();other[held]=False
        predicted.append(predict(x[held],fit(x[other],y[other])));actual.append(y[held]);null.append(y[other].mean(0)[None,:].repeat(len(held),axis=0))
    actual=np.vstack(actual);predicted=np.vstack(predicted);null=np.vstack(null)
    cv_r2=1-np.sum((predicted-actual)**2,axis=0)/np.sum((null-actual)**2,axis=0)
    out.mkdir(exist_ok=True,parents=True)
    with (out/'held-out-robustness.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=comp[0]);w.writeheader();w.writerows(comp)
    result={'scope':'human panel secondary sensitivity; adjustment also removes genuine shared quality/valence variance',
            'training':sum(int(i) for i in train),'nuisance':['intensity','pleasantness','familiarity'],
            'held_out_tests':len(comp),'unchanged':sum(r['same_sign'] for r in comp),'flipped':sum(1-r['same_sign'] for r in comp),
            'family_cv_r2_by_training_quality':dict(zip(dims,map(float,cv_r2)))}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ('family_cv_r2_by_training_quality','scope')})
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stimuli',type=Path);a.add_argument('out',type=Path)
    args=a.parse_args();main(args.stimuli,args.out)
