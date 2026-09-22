#!/usr/bin/env python3
"""Inspect recorded Ultralytics training arguments; metadata is not complete provenance."""
import argparse
import contextlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

RECIPE_KEYS = 'optimizer lr0 lrf momentum weight_decay warmup_epochs epochs batch imgsz box cls dfl close_mosaic mosaic mixup copy_paste scale fliplr flipud degrees shear translate hsv_h hsv_s hsv_v bgr seed patience freeze'.split()


def load_train_args(path):
    if not Path(path).is_file():
        raise ValueError(f'{path}: local checkpoint not found')
    with contextlib.redirect_stdout(sys.stderr):
        from ultralytics import YOLO
        ckpt = YOLO(path).ckpt
    if not isinstance(ckpt, Mapping) or not isinstance(ckpt.get('train_args'), Mapping):
        raise ValueError(f'{path}: no recorded train_args mapping')
    return dict(ckpt['train_args'])


def compare(left, right, keys):
    return {k: {'left_present': k in left, 'right_present': k in right,
                'left': left.get(k), 'right': right.get(k)}
            for k in keys if (k in left) != (k in right) or left.get(k) != right.get(k)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint')
    p.add_argument('--diff')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--keys', nargs='+')
    g.add_argument('--all', action='store_true')
    p.add_argument('--json', action='store_true')
    a = p.parse_args()
    try:
        left = load_train_args(a.checkpoint)
        right = load_train_args(a.diff) if a.diff else None
    except Exception as e:
        p.exit(1, f'{e}\n')
    keys = a.keys if a.keys is not None else (sorted(set(left) | set(right or {})) if a.all or a.diff else [k for k in RECIPE_KEYS if k in left])
    result = compare(left, right, keys) if right is not None else {k: left.get(k) for k in keys}
    if a.json:
        print(json.dumps(result, indent=2, default=str))
    elif right is not None:
        print(f'{len(result)} differing recorded keys')
        for k, v in result.items():
            print(f'{k}: {v}')
        if not result:
            print('Recorded arguments match. Data, initial weights, code and runtime may still differ.')
    else:
        for k in keys:
            print(f'{k} = {left[k] if k in left else "<absent>"}')


if __name__ == '__main__':
    main()
