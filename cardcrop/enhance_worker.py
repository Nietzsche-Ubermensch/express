#!/usr/bin/env python3
"""
Enhance worker v3 — subprocess called by Express.

Pipeline: YOLO detect → orientation → resample → adaptive descratch →
          dual-scale unsharp → CLAHE + vibrance

YOLO weights are optional. If models/card_detector.pt is present the
detector runs and crops to the card; if not, the whole frame is treated
as the card (correct for tight 600dpi scans) and everything else still
works. Nothing fails just because the weights are missing.
"""
import sys, os, json
import cv2
import numpy as np

OUT_W, OUT_H = 1500, 2100
MODEL_PATH = os.environ.get('YOLO_MODEL', os.path.join(os.path.dirname(__file__), 'models', 'card_detector.pt'))

_model = None
_model_tried = False


def get_model():
    """Lazy-load YOLO. Returns None if weights or ultralytics are absent."""
    global _model, _model_tried
    if _model_tried:
        return _model
    _model_tried = True
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        from ultralytics import YOLO
        _model = YOLO(MODEL_PATH)
    except Exception:
        _model = None
    return _model


def detect_card(img):
    """
    YOLO card detection. Returns (x1,y1,x2,y2,conf) or None.

    Crop is only accepted when the box covers >=70% of BOTH dimensions and
    confidence >=0.55. A loose guard here previously let a 0.42-confidence
    box covering 41% of the card through, which stretched a partial crop
    into garbage.
    """
    m = get_model()
    if m is None:
        return None
    try:
        r = m(img, conf=0.35, verbose=False)[0]
        if len(r.boxes) == 0:
            return None
        i = int(r.boxes.conf.argmax())
        x1, y1, x2, y2 = r.boxes[i].xyxy[0].tolist()
        return (int(x1), int(y1), int(x2), int(y2), float(r.boxes[i].conf[0]))
    except Exception:
        return None


def _readability(img_bgr):
    """OCR a downscaled copy, return summed confidence of real words."""
    try:
        import pytesseract
        from PIL import Image
        h, w = img_bgr.shape[:2]
        k = 700.0 / max(h, w)
        small = cv2.resize(img_bgr, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        d = pytesseract.image_to_data(Image.fromarray(gray), config='--psm 6',
                                      output_type=pytesseract.Output.DICT)
        score = 0
        for i in range(len(d['text'])):
            wd = d['text'][i].strip()
            c = int(d['conf'][i])
            if c > 55 and len(wd) >= 3 and any(ch.isalpha() for ch in wd):
                score += c
        return score
    except Exception:
        return 0


def best_orientation(img_bgr):
    """
    Keep the orientation that reads. Landscape-designed card backs (Topps
    patch/relic backs) already read correctly unrotated, so forcing them
    portrait is wrong. Only rotate when a rotation genuinely reads better.
    """
    s_orig = _readability(img_bgr)
    cw = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    ccw = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    s_cw = _readability(cw)
    s_ccw = _readability(ccw)
    best = max(s_orig, s_cw, s_ccw)

    if best < 200:  # image-only front, nothing reads either way
        h, w = img_bgr.shape[:2]
        return (cw, 'cw') if w > h else (img_bgr, 'none')
    if s_orig >= best * 0.85:
        return img_bgr, 'none'
    return (cw, 'cw') if s_cw >= s_ccw else (ccw, 'ccw')


def adaptive_descratch(img):
    """
    Scratch removal with a per-card adaptive threshold.

    A fixed threshold read halftone dots and foil texture as scratches
    (mean 448 per card). Scaling the threshold to each card's own texture
    energy and keeping only long thin components brings that to ~17.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_full = np.abs(cv2.Laplacian(gray, cv2.CV_64F, ksize=3))
    energy = float(np.percentile(lap_full, 92))
    thresh = max(25.0, energy * 2.5)

    mask = (lap_full > thresh).astype(np.uint8) * 255
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    lines = cv2.bitwise_or(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_h),
                           cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_v))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(lines, 8)
    keep = np.zeros_like(lines)
    real = 0
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        aspect = max(w, h) / max(1, min(w, h))
        if area >= 30 and aspect >= 6:
            keep[labels == i] = 255
            real += 1

    if real == 0:
        return img, 0, thresh
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return cv2.inpaint(img, keep, 4, cv2.INPAINT_NS), real, thresh


def enhance(input_path, output_path, opts=None):
    opts = opts or {}
    img = cv2.imread(input_path)
    if img is None:
        return {"error": "cannot read image"}

    src_h, src_w = img.shape[:2]

    # 1. YOLO detect + crop (skipped entirely when weights absent)
    yolo_conf = 0.0
    cropped = False
    det = detect_card(img)
    if det:
        x1, y1, x2, y2, conf = det
        yolo_conf = conf
        bw, bh = x2 - x1, y2 - y1
        if bw >= src_w * 0.70 and bh >= src_h * 0.70 and conf >= 0.55:
            pad = 4
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(src_w, x2 + pad), min(src_h, y2 + pad)
            img = img[y1:y2, x1:x2]
            cropped = True

    # 2. Orientation
    rot = 'none'
    if opts.get('autoRotate', True):
        img, rot = best_orientation(img)

    # 3. Resample, respecting the card's own aspect
    h, w = img.shape[:2]
    if w > h:
        img = cv2.resize(img, (OUT_H, OUT_W), interpolation=cv2.INTER_LANCZOS4)
    else:
        img = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)

    # 4. Descratch
    scratches, thresh = 0, 0.0
    if float(opts.get('descratch', 0.35)) > 0.01:
        img, scratches, thresh = adaptive_descratch(img)

    # 5. Dual-scale unsharp
    sharpen = float(opts.get('sharpen', 0.5))
    if sharpen > 0.01:
        f = img.astype(np.float32)
        img = np.clip(f + (f - cv2.GaussianBlur(f, (3, 3), 1.0)) * (sharpen * 1.7)
                        + (f - cv2.GaussianBlur(f, (5, 5), 2.0)) * (sharpen * 0.8),
                      0, 255).astype(np.uint8)

    # 6. CLAHE + vibrance
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    l = np.clip(l.astype(np.float32) * 1.08, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    if not opts.get('conservative', True):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        s = hsv[:, :, 1]
        hsv[:, :, 1] = np.clip(s * (1.0 + 0.18 * (1.0 - s / 255.0)), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    oh, ow = img.shape[:2]
    return {
        "scratches_found": scratches,
        "scratches_fixed": scratches,
        "scratch_threshold": round(thresh, 1),
        "rotation": rot,
        "yolo_conf": round(yolo_conf, 3),
        "yolo_cropped": cropped,
        "yolo_available": get_model() is not None,
        "source": f"{src_w}x{src_h}",
        "output_size": f"{ow}x{oh}",
        "orientation": "landscape" if ow > oh else "portrait",
    }


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(json.dumps({"error": "usage: enhance_worker.py <input> <output> [opts_json]"}))
        sys.exit(0)
    opts = {}
    if len(sys.argv) > 3:
        try:
            opts = json.loads(sys.argv[3])
        except Exception:
            pass
    print(json.dumps(enhance(sys.argv[1], sys.argv[2], opts), default=str))
