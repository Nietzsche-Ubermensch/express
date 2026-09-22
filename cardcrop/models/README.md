# Wrestling-card detector

No trained weights are included in this repository. The application reports
`weights_missing` until a usable `card_detector.pt` is supplied at `YOLO_MODEL`.
A file existing on disk is not proof of a working detector: the health probe
loads it and runs an inference before reporting `yolo.available: true`.

Previous documentation claimed training on 148 WWE/AEW scans and supplied mAP,
precision and recall figures. Those claims are unverified here: neither the
checkpoint, dataset split, training log nor evaluation output accompanies them.
Do not treat those figures as deployed capability.

Supply a trusted, validated wrestling-card checkpoint through the deployment
source or a persistent mounted path. Record its SHA-256, dataset provenance and
held-out evaluation before publishing accuracy claims. A detector trained only
on tightly cropped scans must be tested on photos before claiming photo support.

Without an accepted detection, enhancement preserves the whole frame, reports
`detection_mode: whole_frame`, and requires a manual crop check. It does not
claim segmentation, perspective rectification, learned upscaling, or that an
uploaded image is a wrestling card. Category verification belongs to the
identification/review step, not the geometric detector.
