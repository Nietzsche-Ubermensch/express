# Wrestling-card detector

`card_detector.pt` is **not stored in this repo**. The Dockerfile downloads it at
build time from `claudepersonal/CardEnhance` (`models/card-seg/v1/best.pt`,
pinned to commit `2521c65`) and verifies SHA-256
`3cd5708b94738d1ba24cdbba73db6b8dbb49d31c903b3c2c69346c87f669a1d6`. A mismatch
fails the build. To change the model, update both `ARG`s in the Dockerfile.

## What it is

YOLO11n detector, one class (`card`), 5,453,274 bytes. Training metadata that
ships with it (`metadata.json` in CardEnhance): 67 train / 17 val images,
6 epochs, imgsz 640, CPU, reported mAP50 0.995 / mAP50-95 0.917.

Those figures are self-reported on a 17-image validation set. Treat them as
a sanity check, not a measured accuracy.

## What was actually verified (2026-09-23)

- Loads under ultralytics 8.4.x and runs inference on CPU.
- On three real scans from the test batch, boxes landed on the card edge:
  0819 conf 0.74, 0996 conf 0.61, 0854 conf 0.45.
- `enhance_worker.py` only accepts a detection with conf >= 0.55 covering
  >= 70% of the frame in both dimensions. 0854 would therefore fall back to
  whole-frame processing, which is correct for tight flatbed scans.

## Runtime behaviour

`/api/health` reports `yolo.available: true` only after the worker has loaded
the model **and** run one inference. If the file is absent or fails to load,
enhancement uses the whole frame and reports `detection_mode: whole_frame`.
