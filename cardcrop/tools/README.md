# Wrestling detector experiments

Run these as separate scripts from cardcrop; they do not start training during the web-service boot.

- `python3 tools/inspect_checkpoint.py models/card_detector.pt --all --json`
- `python3 tools/train_run.py --model yolo26n.pt --data /path/to/data.yaml --dry-run`
- `python3 tools/train_run.py --model yolo26n.pt --data /path/to/data.yaml --epochs 100 --seed 0`
- `python3 tools/eval_matrix.py --weights /path/to/best.pt --data /path/to/data.yaml --out eval.json`
- `python3 tools/ablate.py --model yolo26n.pt --data /path/to/data.yaml --vary imgsz=640,960 --seeds 0,1,2 --dry-run`

Install Ultralytics in a dedicated training environment. Record its version with each run. The production Dockerfile already installs it for optional inference.

Use only checkpoints whose source you trust. Inspection loads the model through Ultralytics. Identical recorded train_args do not establish identical data, initial weights or software.

The evaluation matrix supports detection checkpoints with both heads, not segmentation. `--max-det` is explicit and defaults to 300; it is not asserted to be an immutable architectural limit. See https://docs.ultralytics.com/guides/end2end-detection/ . Per-class AP is emitted only for measured classes. Latency is reported separately for each head on a blank synthetic input, batch 1, full precision; it is not a real-card throughput benchmark.

Datasets must use images/ and labels/ paths, with detection labels `class x y width height`. Nested directories and image-list split files are supported. Missing labels are counted as backgrounds and separately reported. Confirm that they are intentional before evaluating. No fallback to a different split occurs.

The ledger records started/completed/failed events, complete content hashes, actual training arguments and version information. Dry-run needs no torch/Ultralytics installation and neither resolves nor downloads a dataset. Abrupt process termination may leave only a started event.

Ablations keep the first specified variant as the baseline and compare matching successful seeds. Results are descriptive; they do not claim statistical significance. One successful seed never implies zero noise. CSV and summary are saved after each run.

The supplied Sportscardstotest(1).zip contains 424 JPEGs, no annotation files, dataset YAML or checkpoint. It is an image-pipeline fixture, not an independently labelled detector evaluation dataset. No trained card detector is bundled in this change.
