#!/usr/bin/env python3
"""One-variable sweeps with fixed baseline and descriptive paired-seed differences."""
import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path
from run_utils import parse_overrides
from train_run import execute


def paired_summary(rows, variants):
    baseline={r['seed']:r['value'] for r in rows if r['variant']==variants[0] and r['value'] is not None}
    result=[]
    for variant in variants:
        valid={r['seed']:r['value'] for r in rows if r['variant']==variant and r['value'] is not None}
        differences=[valid[s]-baseline[s] for s in sorted(valid.keys() & baseline.keys())]
        result.append({'variant':variant,'n':len(valid),'mean':st.mean(valid.values()) if valid else None,
                       'std':st.stdev(valid.values()) if len(valid)>1 else None,'paired_n':len(differences),
                       'paired_delta_mean':st.mean(differences) if differences else None,
                       'paired_delta_std':st.stdev(differences) if len(differences)>1 else None,
                       'status':'baseline' if variant==variants[0] else ('insufficient_pairs' if len(differences)<2 else 'descriptive_only')})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','data','vary'):p.add_argument('--'+name,required=True)
    p.add_argument('--seeds',default='0,1,2');p.add_argument('--fixed',nargs='*')
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--batch',type=int);p.add_argument('--device');p.add_argument('--metric',default='metrics/mAP50-95(B)')
    p.add_argument('--project',default='ablations');p.add_argument('--out',default='ablations/results.csv')
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    try:
        key,sep,text=a.vary.partition('=')
        if not sep or not key or key in ('seed','project','name','data','model'):raise ValueError('Invalid sweep key')
        values=[parse_overrides([key+'='+v])[key] for v in text.split(',')]
        variants=[json.dumps(v,sort_keys=True) for v in values]
        seeds=[int(s) for s in a.seeds.split(',')]
        if len(set(variants))!=len(values) or len(set(seeds))!=len(seeds):raise ValueError('Duplicate variants or seeds')
        fixed=parse_overrides(a.fixed)
        if set(fixed)&{key,'seed','project','name','data','model'}:raise ValueError('Fixed arguments conflict with controlled sweep settings')
    except ValueError as e:p.error(str(e))
    base={'data':a.data,'epochs':a.epochs,'imgsz':a.imgsz,**fixed}
    if a.batch is not None:base['batch']=a.batch
    if a.device is not None:base['device']=a.device
    if a.dry_run:
        print(json.dumps([{**base,key:v,'seed':s} for v in values for s in seeds],indent=2));return
    rows=[];out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    for i,v in enumerate(values):
        for seed in seeds:
            row={'variant':variants[i],'seed':seed,'value':None,'error':''}
            try:
                result=execute(a.model,{**base,key:v,'seed':seed,'project':a.project,'name':f'{key}_{i}_seed{seed}'},str(out.with_suffix('.ledger.jsonl')))
                value=float(result['metrics'][a.metric])
                if not math.isfinite(value):raise ValueError('Nonfinite metric')
                row['value']=value
            except Exception as e:row['error']=str(e)
            rows.append(row)
            with out.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['variant','seed','value','error']);w.writeheader();w.writerows(rows)
            out.with_suffix('.summary.json').write_text(json.dumps({'baseline':variants[0],'metric':a.metric,'summary':paired_summary(rows,variants)},indent=2))
    raise SystemExit(1 if any(r['error'] for r in rows) else 0)


if __name__=='__main__':main()
