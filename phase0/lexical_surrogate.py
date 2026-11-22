"""Paired leave-two-donors-out lexical writer on historical Costco hidden states.
The predicted vectors are never written to a model. Compare their geometric
alignment with the already archived patched state. No new Kimi inference.
"""
from pathlib import Path
from itertools import combinations
import argparse
import hashlib
import json
import csv
import numpy as np

def kernel(texts):
    # Character features, fixed normalized binary counts. No neural representation
    # or model answer enters the input to the surrogate writer.
    docs=[]
    for text in texts:
        t=" "+text.lower()+" ";docs.append({t[i:i+n] for n in (3,4,5) for i in range(len(t)-n+1)})
    return np.array([[len(a & b)/(len(a)*len(b))**.5 for b in docs] for a in docs])

def train_weights(K,tr,te,G):
    k=K[np.ix_(tr,tr)]
    m=len(tr)
    c=np.eye(m)-np.ones((m,m))/m
    # Kernel centering is learned exclusively on the training sentences.
    kc=k-k.mean(axis=0)-k.mean(axis=1,keepdims=True)+k.mean()
    cross=K[np.ix_(te,tr)]
    cross=cross-k.mean(axis=0)
    cross=cross-cross.mean(axis=1,keepdims=True)
    base=np.linalg.solve(kc+np.eye(m),cross.T).T
    gc=G-G.mean(axis=0)-G.mean(axis=1,keepdims=True)+G.mean()
    ev,u=np.linalg.eigh(gc)
    u=u[:,-3:]
    return base,u,c

def main(src,out,permutations):
    meta=json.loads((src/'native-rows.json').read_text());results=[];groups=[]
    for stage,record in meta['stages'].items():
        rows=record['rows'];text=[r['donor_text'] for r in rows];n=len(rows);K=kernel(text)
        with np.load(src/record['file'],allow_pickle=False) as z:
            D=(z['donor'].astype('float64')-z['dots'].astype('float64'))
            P=(z['patched'].astype('float64')-z['dots'].astype('float64'))
        for li,L in enumerate(meta['layers']):
            gg=np.einsum('ik,jk->ij',D[:,li],D[:,li],optimize=False)
            pg=np.einsum('ik,jk->ij',P[:,li],D[:,li],optimize=False)
            pn=np.linalg.norm(P[:,li],axis=1)
            for i,j in combinations(range(n),2):
                te=np.array([i,j]);tr=np.array([k for k in range(n) if k not in (i,j)])
                g=gg[np.ix_(tr,tr)];z=pg[np.ix_(te,tr)]
                base,u,c=train_weights(K,tr,te,g)
                rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f"air/{stage}/{i}/{j}".encode()).digest()[:4],'big'))
                perm=np.array([np.arange(len(tr))]+[rng.permutation(len(tr)) for _ in range(permutations)])
                # If fit labels are permuted, the nonzero output-PC subspace is
                # identical. The training label/covariate association changes.
                baseline_center=base-base.mean(axis=1,keepdims=True)
                centered=baseline_center[:,np.argsort(perm,axis=1)].transpose(1,0,2)
                weights=np.einsum('pik,ok->pio',np.einsum('pim,mk->pik',centered,u,optimize=False),u,optimize=False)+1/len(tr)
                num=np.einsum('im,pjm->pij',z,weights,optimize=False)
                denom=np.einsum('pim,mn,pin->pi',weights,g,weights,optimize=False)
                if np.min(denom)<=0:raise ValueError("Degenerate predicted direction")
                sim=num/(pn[te][None,:,None]*np.sqrt(denom)[:,None,:])
                if not np.isfinite(sim).all():raise ValueError("Nonfinite geometric score")
                for a,(selfidx,other) in enumerate(((i,j),(j,i))):
                    margin=sim[:,a,a]-sim[:,a,1-a]
                    score=lambda x: np.where(x>1e-12,1.,np.where(x< -1e-12,0.,.5))
                    results.append(dict(stage=stage,layer=L,target=rows[selfidx]['donor_key'],foil=rows[other]['donor_key'],
                        same_condition=rows[selfidx]['condition']==rows[other]['condition'],
                        same_length=rows[selfidx]['slot_rows']==rows[other]['slot_rows'],
                        lexical_correct=float(score(margin[0])),permuted_correct=float(score(margin[1:]).mean()),
                        margin=float(margin[0]),permuted_margin=float(margin[1:].mean())))
            subset=[r for r in results if r['stage']==stage and r['layer']==L]
            for kind,ss in [('all',subset),('same_condition',[r for r in subset if r['same_condition']]),
                            ('same_length_and_condition',[r for r in subset if r['same_condition'] and r['same_length']])]:
                groups.append(dict(stage=stage,layer=L,subset=kind,directed_pairs=len(ss),unique_targets=len({r['target'] for r in ss}),
                    correct=sum(r['lexical_correct'] for r in ss)/len(ss),
                    shuffled=sum(r['permuted_correct'] for r in ss)/len(ss)))
    out.mkdir(exist_ok=True,parents=True)
    with (out/'directed-comparisons.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=results[0]);w.writeheader();w.writerows(results)
    (out/'summary.json').write_text(json.dumps(dict(rank=3,penalty=1,shuffle_comparisons_per_pair=permutations,results=groups),indent=2)+'\n')
    print([r for r in groups if r['stage']=='k1' and r['layer']==71])

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('source',type=Path);a.add_argument('output',type=Path);a.add_argument('--permutations',type=int,default=64)
    opts=a.parse_args();main(opts.source,opts.output,opts.permutations)
