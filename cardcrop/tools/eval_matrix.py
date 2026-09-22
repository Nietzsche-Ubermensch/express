#!/usr/bin/env python3
"""Evaluate both detection heads with explicit max_det and per-head latency."""
import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path
from run_utils import resolve, split_images, label_path, file_sha


def instance_density(data, split, max_det=300):
    counts, missing = [], 0
    for image in split_images(resolve(data), split):
        label = label_path(image)
        if not label.exists():
            missing += 1
            counts.append(0)
            continue
        lines = [line.split() for line in label.read_text().splitlines() if line.strip()]
        for line in lines:
            if len(line) != 5:
                raise ValueError(f'{label}: expected detection labels: class x y w h')
            values = list(map(float, line))
            if not all(math.isfinite(v) for v in values) or values[0] < 0 or not values[0].is_integer() or not all(0 <= v <= 1 for v in values[1:]):
                raise ValueError(f'{label}: invalid detection label')
        counts.append(len(lines))
    return {'images':len(counts), 'missing_labels':missing, 'max_instances':max(counts),
            'median_instances':statistics.median(counts), 'images_over_max_det':sum(c > max_det for c in counts),
            'max_det':max_det, 'note':'Missing labels counted as backgrounds; confirm they are intentionally unlabelled.'}


def per_class_metrics(box, names):
    indices = getattr(box, 'ap_class_index', None)
    if indices is None:
        return []
    rows = []
    for i in indices:
        i = int(i)
        name = names.get(i, str(i)) if isinstance(names, dict) else names[i]
        rows.append({'class_id':i,'name':name,'ap50_95':float(box.maps[i])})
    return sorted(rows, key=lambda x:x['ap50_95'])


def check_head(model, end2end):
    if getattr(model, 'task', None) != 'detect':
        raise ValueError('This evaluation matrix supports detection checkpoints only')
    head = model.model.model[-1]
    if not getattr(head, 'end2end', False):
        raise ValueError('Checkpoint has no end-to-end detection head; dual-head comparison unsupported')


def run_val(weights, data, imgsz, batch, device, end2end, split, max_det=300):
    from ultralytics import YOLO
    m = YOLO(weights)
    check_head(m, end2end)
    r = m.val(data=data,imgsz=imgsz,batch=batch,device=device,split=split,end2end=end2end,
              max_det=max_det,verbose=False,plots=False)
    return {'end2end':end2end,'max_det':max_det,'map50_95':float(r.box.map),'map50':float(r.box.map50),
            'map75':float(r.box.map75),'speed_ms':dict(r.speed),
            'per_class_worst_first':per_class_metrics(r.box,r.names)}


def latency(weights, imgsz, device, n, warmup, end2end, max_det):
    import numpy as np
    import torch
    from ultralytics import YOLO
    if n < 1 or warmup < 0:
        raise ValueError('latency-n must be positive; warmup cannot be negative')
    m = YOLO(weights)
    check_head(m,end2end)
    img = np.zeros((imgsz,imgsz,3),dtype=np.uint8)
    def predict():
        m.predict(img,imgsz=imgsz,device=device,end2end=end2end,max_det=max_det,verbose=False,half=False)
    predict()  # initialize predictor outside measurement
    actual = m.predictor.device
    def sync():
        if actual.type == 'cuda': torch.cuda.synchronize(actual)
        elif actual.type == 'mps': torch.mps.synchronize()
    for _ in range(warmup): predict()
    samples = []
    for _ in range(n):
        sync(); t0 = time.perf_counter(); predict(); sync()
        samples.append((time.perf_counter()-t0)*1000)
    return {'n':n,'warmup':warmup,'initialization_runs':1,'device':str(actual),'batch':1,'half':False,
            'imgsz':imgsz,'end2end':end2end,'max_det':max_det,'platform':platform.platform(),
            'hardware':torch.cuda.get_device_name(actual) if actual.type=='cuda' else platform.processor(),
            **{f'p{q}_ms':float(np.percentile(samples,q)) for q in (50,90,99)},
            'mean_ms':statistics.mean(samples),'max_ms':max(samples),
            'note':'Synthetic blank image; predict wall clock including pre/postprocessing; not representative card latency.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',required=True);p.add_argument('--data');p.add_argument('--device')
    p.add_argument('--split',default='val');p.add_argument('--out',default='eval.json')
    for name,default in [('imgsz',640),('batch',8),('max-det',300),('latency-n',50),('latency-warmup',10),('top-worst',10)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--latency-only',action='store_true');p.add_argument('--no-latency',action='store_true')
    a=p.parse_args()
    if not a.data and not a.latency_only: p.error('--data required unless --latency-only')
    if a.latency_only and a.no_latency: p.error('Conflicting latency flags')
    if min(a.imgsz,a.batch,a.max_det,a.latency_n,a.top_worst)<1 or a.latency_warmup<0: p.error('Invalid numeric argument')
    if not Path(a.weights).is_file(): p.error('A local checkpoint is required')
    report={'weights':a.weights,'weights_sha256':file_sha(a.weights),'data':a.data,'split':a.split,'heads':{},'latency':{}}
    failed=False
    for flag in (True,False):
        label=f'end2end_{str(flag).lower()}'
        if not a.latency_only:
            try: report['heads'][label]=run_val(a.weights,a.data,a.imgsz,a.batch,a.device,flag,a.split,a.max_det)
            except Exception as e: report['heads'][label]={'error':str(e)};failed=True
        if not a.no_latency:
            try: report['latency'][label]=latency(a.weights,a.imgsz,a.device,a.latency_n,a.latency_warmup,flag,a.max_det)
            except Exception as e: report['latency'][label]={'error':str(e)};failed=True
    if not a.latency_only:
        try: report['instance_density']=instance_density(a.data,a.split,a.max_det)
        except Exception as e: report['instance_density']={'error':str(e)};failed=True
    report['status']='failed' if failed else 'completed'
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,default=str,allow_nan=False))
    print(json.dumps(report,indent=2,default=str))
    raise SystemExit(1 if failed else 0)


if __name__=='__main__': main()
