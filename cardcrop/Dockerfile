FROM node:20-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY package.json ./
RUN npm install --omit=dev

# CPU-only torch keeps the image ~2GB instead of ~6GB with CUDA.
# Railway has no GPU, so the CUDA wheels would be dead weight.
RUN pip3 install --no-cache-dir --break-system-packages \
      --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip3 install --no-cache-dir --break-system-packages \
      opencv-python-headless numpy Pillow pytesseract ultralytics

COPY . .

# models/ holds card_detector.pt (YOLO weights). Optional — enhance_worker.py
# falls back to whole-frame processing when it is absent.
RUN mkdir -p /app/models /data/uploads /data/enhanced

ENV NODE_ENV=production \
    STORAGE_DIR=/data \
    YOLO_MODEL=/app/models/card_detector.pt \
    YOLO_CONFIG_DIR=/tmp/ultralytics

EXPOSE 8080
CMD ["node", "server.js"]
