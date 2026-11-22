"""Cross-object/cross-wording decode of old stated modality; no inference or writes."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

def auc(y,s):
    y=np.asarray(y,dtype=bool);n=int(y.sum());m=len(y)-n
    if n*m==0:raise ValueError("Unbalanced test fold");
    return float((rankdata(s)[y].sum()-n*(n+1)/2)/(n*m))

def chargrams(text):
    text=" "+text.lower()+" "
    return Counter(text[i:i+n] for n in (3,4,5) for i in range(len(text)-n+1))

def lexical(text,train):
    counts=[chargrams(x) for x in text]
    vocab=sorted(set().union(*[c.keys() for i,c in enumerate(counts) if train[i]]))
    cols={s:i for i,s in enumerate(vocab)}
    X=np.zeros((len(text),len(cols)),dtype=np.float64)
    for i,c in enumerate(counts):
        for name,count in c.items():
            if name in cols:X[i,cols[name]]=1+np.log(count)
    idf=1+np.log((1+sum(train))/(1+(X[train]>0).sum(0)))
    X*=idf
    return X/np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-10)

def score(features,source,y,train,test):
    # Each class/source cell contributes one centroid, irrespective of template frequency.
    means=[np.asarray([features[train&(source==obj)&(y==label)].mean(axis=0)
                       for obj in np.unique(source[train])]).mean(axis=0) for label in (False,True)]
    d=means[1]-means[0]; midpoint=(means[1]+means[0])/2
    s=np.einsum("ij,j->i",features[test]-midpoint,d,optimize=False)
    return s

def run(directory,output):
    if (directory/'frame-records.json').is_file():
        meta=json.loads((directory/'frame-records.json').read_text())
        items=meta['frame_records'];ix=list(range(len(items)));source_names=sorted({r['object'] for r in items})
        with np.load(directory/'frame-states.npz',allow_pickle=False) as z:
            states={l:z['attn_in_end_of_msg'][j].astype('float64') for j,l in enumerate(meta['layers'])}
    else:
        meta=json.loads((directory/'activations-s1.json').read_text());q=json.loads((directory/'prompts.json').read_text())['items']
        bykey={x['key']:x for x in q};q=[bykey[k] for k in meta['keys']]
        source_names=sorted({x['object'] for x in q if x['condition']=='smell'}&
                            {x['object'] for x in q if x['condition']=='see'})
        ix=[i for i,x in enumerate(q) if x['object'] in source_names and x['condition'] in ('smell','see')]
        items=[q[i] for i in ix]
        raw=np.memmap(directory/'activations-s1.f16',mode='r',dtype='float16',shape=tuple(meta['shape']))
        pos=meta['positions'].index('end_of_msg');site=meta['sites'].index('attn_in')
        states={l:raw[site,meta['layers'].index(l),ix,pos,:].astype('float64') for l in (23,47,71)}
    source=np.array([x['object'] for x in items]);y=np.array([x['condition']=='smell' for x in items])
    frames=[x['text'].lower().replace(x['phrase'].lower(),'{source}') for x in items]
    assert all('{source}' in s for s in frames)
    groups={}
    for label in (False,True):
        unique=sorted({s for i,s in enumerate(frames) if y[i]==label}, key=lambda s:hashlib.sha256(s.encode()).hexdigest())
        for i,s in enumerate(unique):groups[s]=i%2
    fold=np.array([groups[k] for k in frames])
    scores=[];rows=[]
    for hold in source_names:
        for f in (0,1):
            train=(source!=hold)&(fold!=f);test=(source==hold)&(fold==f)
            assert not(set(np.array(frames)[train])&set(np.array(frames)[test]))
            text_features=lexical([x['text'] for x in items],train)
            ff={'text-char-ngrams':text_features}
            for l in (23,47,71):
                ff[f'native-L{l}']=states[l]
            for feature,X in ff.items():
                s=score(X,source,y,train,test)
                scores.append(dict(held_out_source=hold,held_out_frame_fold=f,feature=feature,train_n=int(train.sum()),
                                   test_n=int(test.sum()),n_smell=int(y[test].sum()),auc=auc(y[test],s),
                                   accuracy=float(((s>0)==y[test]).mean())))
                for i,v in zip(np.flatnonzero(test),s):
                    rows.append(dict(key=items[i]['key'],source=hold,frame=frames[i],label=items[i]['condition'],
                                     fold=f,feature=feature,margin=float(v)))
    output.mkdir(exist_ok=True,parents=True)
    for name,data in [('tests.csv',scores),('predictions.csv',rows)]:
        with (output/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
    agg={k:{'macro_auc':float(np.mean([x['auc'] for x in scores if x['feature']==k])),
            'range_auc':[float(fn([x['auc'] for x in scores if x['feature']==k])) for fn in (min,max)],
            'macro_accuracy':float(np.mean([x['accuracy'] for x in scores if x['feature']==k]))}
         for k in sorted({x['feature'] for x in scores})}
    (output/'summary.json').write_text(json.dumps({'prompts':len(ix),'objects':source_names,
        'distinct_normalized_templates':len(set(frames)),'feature_results':agg,
        'scope':'Unmodified archival representations from explicitly worded odor/sight prompts; no new model calls.'},indent=2)+'\n')
    print(agg)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
    run(a.source,a.output)
