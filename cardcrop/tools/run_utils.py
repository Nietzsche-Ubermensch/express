"""Shared dataset resolution and complete content provenance."""
import hashlib
import json
import math
from pathlib import Path

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def file_sha(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_overrides(pairs):
    out = {}
    for item in pairs or []:
        k, sep, v = item.partition('=')
        if not sep or not k.strip() or not v.strip() or k in out:
            raise ValueError(f'Expected unique key=value, got {item!r}')
        try:
            value = json.loads(v)
        except ValueError:
            value = None if v.lower() in ('none', 'null') else v
            if v.lower() in ('true', 'false'):
                value = v.lower() == 'true'
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'Nonfinite option: {item}')
        out[k] = value
    return out


def split_images(dataset, split):
    if not dataset.get(split):
        raise ValueError(f'Missing requested dataset split: {split}')
    roots = dataset[split]
    roots = [roots] if isinstance(roots, (str, Path)) else roots
    images = set()
    for root in roots:
        p = Path(root)
        if p.is_dir():
            candidates = p.rglob('*')
        elif p.is_file() and p.suffix.lower() == '.txt':
            candidates = [p.parent / line.strip()[2:] if line.strip().startswith('./') else Path(line.strip())
                          for line in p.read_text().splitlines() if line.strip()]
        elif p.is_file():
            candidates = [p]
        else:
            raise ValueError(f'Dataset path missing: {p}')
        for f in candidates:
            if f.suffix.lower() in EXTENSIONS:
                if not f.is_file():
                    raise ValueError(f'Image missing: {f}')
                images.add(f.resolve())
    if not images:
        raise ValueError(f'No images in split {split}')
    return sorted(images)


def label_path(image):
    parts = list(image.parts)
    if 'images' not in parts:
        raise ValueError(f'Image path has no images directory: {image}')
    index = len(parts) - 1 - parts[::-1].index('images')
    parts[index] = 'labels'
    return Path(*parts).with_suffix('.txt')


def resolve(data):
    from ultralytics.data.utils import check_det_dataset
    return check_det_dataset(data, autodownload=False)


def dataset_fingerprint(data):
    d = resolve(data)
    manifest = []
    for split in ('train', 'val'):
        for image in split_images(d, split):
            label = label_path(image)
            manifest.append({'split': split, 'image': str(image), 'image_sha256': file_sha(image),
                             'label': str(label), 'label_sha256': file_sha(label)})
    return {'yaml': str(data), 'yaml_sha256': file_sha(data), 'images': len(manifest),
            'manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
            'manifest': manifest}
