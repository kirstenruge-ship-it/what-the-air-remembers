"""Monte Carlo operating characteristics of the registered 12-family CI rules.
Synthetic paired effects only; ratios in this file are not Kimi results.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def evaluate(out):
    rng=np.random.default_rng(927131)
    n=12
    # Families are the sampling unit; each illustrative value already averages the
    # within-family order/contrast replicates. Reuse bootstrap weights across
    # studies for precision and deterministic reproduction.
    weights=rng.multinomial(n,np.full(n,1/n),size=2000)/n
    num_studies=6000; rows=[]
    for sd in [.2,.5,1.]:
        for truth in [0.,.15,.3,.6,.9]:
            for shape in ("gaussian","heavy_tail"):
                outcomes=[0,0,0,0]
                widths=[]
                for begin in range(0,num_studies,300):
                    e=(rng.normal(size=(300,n)) if shape=='gaussian' else rng.standard_t(4,size=(300,n))/np.sqrt(2))*sd+truth
                    bs=np.einsum("ik,jk->ij",e,weights,optimize=False)
                    if not np.isfinite(bs).all():raise ValueError("Nonfinite bootstrap mean")
                    lo,hi=np.quantile(bs,[.025,.975],axis=1)
                    outcomes[0]+=int((lo>.3).sum())
                    outcomes[1]+=int(((lo>=-.3)&(hi<=.3)).sum())
                    outcomes[2]+=int((hi<-.3).sum())
                    outcomes[3]+=int(((lo<=truth)&(hi>=truth)).sum())
                    widths.extend(hi-lo)
                rows.append(dict(n_families=n,truth=truth,sd=sd,shape=shape,n_synthetic=num_studies,
                                 lower_above_margin=outcomes[0]/num_studies,
                                 practical_null=outcomes[1]/num_studies,upper_below_negative_margin=outcomes[2]/num_studies,
                                 interval_covers_truth=outcomes[3]/num_studies,median_width=float(np.median(widths))))
    out.mkdir(parents=True,exist_ok=True)
    with (out/'operating-characteristics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (out/'design.json').write_text(json.dumps({"scope":"synthetic sensitivity of CI alone; control, safety and qualification gates can only reduce rate of superiority claims",
        "families":n,"bootstrap_draws":len(weights),"studies_per_cell":num_studies,"seed":927131,"superiority_margin":.30,
        "minimum_exact_two_sided_sign_flip_p_n4":2/16},indent=2)+'\n')
    for row in rows:
        if row['shape']=='gaussian' and row['sd']==.5 and row['truth'] in (0.,.3,.6):print(row)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);args=ap.parse_args();evaluate(args.out)
