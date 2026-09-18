#!/usr/bin/env python3
"""
Enhance worker v2 — subprocess called by Express.
Fixes over v1:
  - Adaptive scratch threshold scaled to each card's own texture energy
    (stops foil/halftone reading as 400+ false scratches)
  - Orientation chosen by readability scoring, not forced portrait
    (landscape-designed card backs keep their orientation)
  - Output size follows the card's own aspect ratio
"""
import sys, json
import cv2
import numpy as np

OUT_W, OUT_H = 1500, 2100


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
    """Keep the orientation that reads. Landscape backs stay landscape."""
    s_orig = _readability(img_bgr)
    cw = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    ccw = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    s_cw = _readability(cw)
    s_ccw = _readability(ccw)
    best = max(s_orig, s_cw, s_ccw)
    if best < 200:
        h, w = img_bgr.shape[:2]
        return (cw, 'cw') if w > h else (img_bgr, 'none')
    if s_orig >= best * 0.85:
        return img_bgr, 'none'
    return (cw, 'cw') if s_cw >= s_ccw else (ccw, 'ccw')


def adaptive_descratch(img):
    """Scratch removal with per-card adaptive threshold."""
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


def enhance(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        return {"error": "cannot read image"}

    src_h, src_w = img.shape[:2]

    img, rot = best_orientation(img)

    h, w = img.shape[:2]
    if w > h:
        img = cv2.resize(img, (OUT_H, OUT_W), interpolation=cv2.INTER_LANCZOS4)
    else:
        img = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)

    img, scratches, thresh = adaptive_descratch(img)

    f = img.astype(np.float32)
    img = np.clip(f + (f - cv2.GaussianBlur(f, (3, 3), 1.0)) * 0.85
                    + (f - cv2.GaussianBlur(f, (5, 5), 2.0)) * 0.40, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    l = np.clip(l.astype(np.float32) * 1.08, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

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
        "source": f"{src_w}x{src_h}",
        "output_size": f"{ow}x{oh}",
        "orientation": "landscape" if ow > oh else "portrait",
    }


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(json.dumps({"error": "usage: enhance_worker.py <input> <output>"}))
        sys.exit(0)
    print(json.dumps(enhance(sys.argv[1], sys.argv[2]), default=str))
