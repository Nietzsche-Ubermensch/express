#!/usr/bin/env python3
import sys, json
import cv2
import numpy as np

def enhance(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None: return {"error": "cannot read image"}
    h, w = img.shape[:2]
    if w > h: img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE); h, w = img.shape[:2]
    img = cv2.resize(img, (1500, 2100), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap_norm = (np.abs(lap) / (np.abs(lap).max() + 1e-6) * 255).astype(np.uint8)
    _, scratch_mask = cv2.threshold(lap_norm, 30, 255, cv2.THRESH_BINARY)
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    lines = cv2.bitwise_or(cv2.morphologyEx(scratch_mask, cv2.MORPH_OPEN, kh), cv2.morphologyEx(scratch_mask, cv2.MORPH_OPEN, kv))
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    num_labels, _ = cv2.connectedComponents(lines)
    scratches = max(0, num_labels - 1)
    if scratches > 0: img = cv2.inpaint(img, lines, 5, cv2.INPAINT_NS)
    f = img.astype(np.float32)
    img = np.clip(f + (f - cv2.GaussianBlur(f, (3,3), 1.0)) * 0.85 + (f - cv2.GaussianBlur(f, (5,5), 2.0)) * 0.40, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    l = np.clip(l.astype(np.float32) * 1.11, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return {"scratches_found": scratches, "scratches_fixed": scratches}

if __name__ == '__main__':
    if len(sys.argv) < 3: print(json.dumps({"error": "usage: enhance_worker.py <input> <output>"})); sys.exit(0)
    print(json.dumps(enhance(sys.argv[1], sys.argv[2]), default=str))
