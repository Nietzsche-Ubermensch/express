#!/usr/bin/env python3
"""Train with full-file hashes and a ledger that records failures as well as success."""
import argparse
import json
import platform
import subprocess
import time
import uuid
from pathlib import Path
from run_utils import file_sha, dataset_fingerprint, parse_overrides


def git(args):
    r = subprocess.run(['git', *args], capture_output=True, text=True, timeout=15)
    return r.stdout.strip() if r.returncode == 0 else None


def append(ledger, record):
    p = Path(ledger)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as f:
        f.write(json.dumps(record, default=str, allow_nan=False) + '\n')
        f.flush()


def execute(model_path, cfg, ledger, tag=''):
    run_id = uuid.uuid4().hex
    record = {'run_id': run_id, 'tag': tag, 'model': model_path, 'config': cfg,
              'status': 'started', 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    append(ledger, record)
    t0 = time.perf_counter()
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
        record.update(dataset=dataset_fingerprint(cfg['data']), git_sha=git(['rev-parse', 'HEAD']),
                      git_status=git(['status', '--porcelain']),
                      versions={'ultralytics': ultralytics.__version__, 'torch': torch.__version__, 'python': platform.python_version()},
                      platform=platform.platform(), cuda_available=torch.cuda.is_available())
        model = YOLO(model_path)
        resolved = getattr(model, 'ckpt_path', None) or model_path
        record['initial_weights_sha256'] = file_sha(resolved)
        result = model.train(**cfg)
        record['metrics'] = {k: float(v) for k, v in result.results_dict.items()}
        trainer = model.trainer
        record['resolved_train_args'] = vars(trainer.args)
        best = Path(trainer.save_dir) / 'weights/best.pt'
        record.update(status='completed', save_dir=str(trainer.save_dir), best_weights=str(best), best_weights_sha256=file_sha(best))
        return record
    except BaseException as e:
        record.update(status='failed', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        record['duration_s'] = time.perf_counter() - t0
        record['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        append(ledger, record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    for name, default in [('epochs', 100), ('imgsz', 640), ('seed', 0)]:
        p.add_argument('--'+name, type=int, default=default)
    for name in ('batch', 'patience'):
        p.add_argument('--'+name, type=int)
    for name in ('device', 'optimizer', 'project', 'name'):
        p.add_argument('--'+name)
    p.add_argument('--lr0', type=float)
    p.add_argument('--tag', default='')
    p.add_argument('--set', nargs='*')
    p.add_argument('--ledger', default='runs/ledger.jsonl')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    cfg = {k: getattr(a,k) for k in ('data','epochs','imgsz','seed','batch','patience','device','optimizer','project','name','lr0') if getattr(a,k) is not None}
    try:
        cfg.update(parse_overrides(a.set))
        if int(cfg['epochs']) <= 0 or int(cfg['imgsz']) <= 0:
            raise ValueError('epochs and imgsz must be positive')
    except (ValueError, TypeError) as e:
        p.error(str(e))
    if a.dry_run:
        print(json.dumps({'model':a.model, 'config':cfg, 'training_started':False}, indent=2))
        return
    execute(a.model, cfg, a.ledger, a.tag)


if __name__ == '__main__':
    main()
