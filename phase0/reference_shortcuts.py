"""External human-label shortcut checks on a frozen held-out cohort; no Kimi draws."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
import argparse

def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def sign(f):return 1 if f>1e-12 else -1 if f<-1e-12 else 0

def run(root,out):
    st=root/'stimuli'
    roster={str(r['cid']):r for r in json.loads((st/'roster.json').read_text())}
    human=rows(st/'human_choices.csv');nuisance={tuple(sorted((r['cid_i'],r['cid_j']))):r for r in rows(st/'nuisance.csv')}
    reliability={(r['cid'],r['pair_id']):float(r['split_half_sign_agreement']) for r in rows(root/'analysis/phase0-human/human-reliability-pairs.csv')}
    h_by_pair=defaultdict(list)
    for r in human:
        if roster[r['cid']]['split']=='fit':h_by_pair[r['pair_id']].append(sign(float(r['human_margin'])))
    allrows=[]
    for r in human:
        cid,pair=r['cid'],r['pair_id']
        if (cid,pair) not in reliability:continue
        left=nuisance[tuple(sorted((cid,r['left_cid'])))];right=nuisance[tuple(sorted((cid,r['right_cid'])))]
        target=sign(float(r['human_margin']));assert target!=0
        experiments={'training_majority':sign(sum(h_by_pair[pair]))}
        for metric in ('name_similarity','fingerprint_similarity','intensity_difference','pleasantness_difference','familiarity_difference'):
            s=float(left[metric])-float(right[metric])
            experiments[metric]=sign(s) if metric.endswith('similarity') else -sign(s)
        for kind,pred in experiments.items():
            allrows.append(dict(cid=cid,pair_id=pair,baseline=kind,prediction=pred,truth=target,
                                correct=(1 if pred==target else .5 if pred==0 else 0),
                                reliability=reliability[cid,pair]))
    out.mkdir(exist_ok=True,parents=True)
    with (out/'comparisons.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    agg=[]
    for kind in sorted(set(r['baseline'] for r in allrows)):
        for cut in (0.,.8):
            selected=[r for r in allrows if r['baseline']==kind and r['reliability']>=cut]
            agg.append({'baseline':kind,'reliability_floor':cut,'n':len(selected),'ties':sum(r['prediction']==0 for r in selected),
                        'correct_half_credit':sum(r['correct'] for r in selected),'accuracy':mean(r['correct'] for r in selected)})
    labels=[]
    for ref in sorted({r['pair_id'] for r in allrows}):
        selected=[r for r in allrows if r['pair_id']==ref and r['baseline']=='training_majority']
        labels.append({'pair':ref,'training_left':sum(x==1 for x in h_by_pair[ref]),'training_total':len(h_by_pair[ref]),
                       'held_left':sum(x['truth']==1 for x in selected),'held_total':len(selected)})
    (out/'summary.json').write_text(json.dumps({'scope':'Frozen external human benchmark. A/B orientations are canonical reference identity, not displayed letter. No Kimi data.','scores':agg,'class_balance':labels},indent=2)+'\n')
    print(agg);print(labels)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.root,a.output)
