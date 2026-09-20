# YOLO weights

`card_detector.pt` goes here. The file is 22MB, too large to commit through
the GitHub API, so add it directly:

```bash
git lfs install
git lfs track "cardcrop/models/*.pt"
cp /path/to/card_detector_best.pt cardcrop/models/card_detector.pt
git add .gitattributes cardcrop/models/card_detector.pt
git commit -m "add trained YOLO card detector"
git push
```

Without LFS, a plain `git add` of a 22MB binary also works — GitHub's hard
limit is 100MB per file. LFS just keeps the repo lean.

## What it is

YOLOv11n-seg fine-tuned on 148 WWE/AEW card scans.

| Metric | Value |
|---|---|
| Box mAP50 | 0.956 |
| Box mAP50-95 | 0.841 |
| Mask mAP50 | 0.957 |
| Precision | 0.953 |
| Recall | 0.931 |

Trained 25 epochs, imgsz 320, batch 4, on CPU. Best epoch was 10.

## Verifying it loaded

`/api/health` reports `yolo.available`. Per-card, the enhance response
includes `yolo_available`, `yolo_conf` and `yolo_cropped`.

With no weights present the app runs normally — `enhance_worker.py` treats
the whole frame as the card, which is correct for tight 600dpi scans and
wrong only for photos with visible background.

## Retraining

The dataset generator lives in the batch pipeline. It writes YOLO-format
segmentation labels from CV detections, splits train/val, and emits a
`data.yaml`:

```bash
yolo segment train data=<dataset>/data.yaml model=yolo11n-seg.pt \
  epochs=50 imgsz=640 batch=8 lr0=0.001
```

Add phone photos and binder-page scans before retraining — the current
weights only ever saw tight flatbed scans, so they learned
"card = full frame" and will not generalise to visible backgrounds.
