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

# Upgrade Debian pip so wheel metadata names are normalized correctly.
RUN pip3 install --no-cache-dir --break-system-packages --upgrade pip

# Explicit CPU versions prevent CUDA wheels; PyPI supplies general dependencies.
RUN pip3 install --no-cache-dir --break-system-packages \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      torch==2.5.1+cpu torchvision==0.20.1+cpu
RUN pip3 install --no-cache-dir --break-system-packages \
      opencv-python-headless numpy Pillow pytesseract ultralytics

COPY . .

RUN mkdir -p /app/models /data/uploads /data/enhanced

# YOLO card detector (yolo11n, single class "card"), fetched at build time from
# claudepersonal/CardEnhance pinned to a commit and verified by SHA-256, so the
# binary never has to live in this repo. A checksum mismatch or failed download
# fails the build loudly rather than shipping a wrong or missing model.
# enhance_worker.py still falls back to whole-frame processing if the model
# cannot load at runtime; /api/health reports yolo.available either way.
ARG CARD_DETECTOR_URL=https://raw.githubusercontent.com/claudepersonal/CardEnhance/2521c65085e3640d4f16a194db991305637e64ac/models/card-seg/v1/best.pt
ARG CARD_DETECTOR_SHA256=3cd5708b94738d1ba24cdbba73db6b8dbb49d31c903b3c2c69346c87f669a1d6
RUN python3 -c "import hashlib,sys,urllib.request; d=urllib.request.urlopen(sys.argv[1],timeout=120).read(); h=hashlib.sha256(d).hexdigest(); assert h==sys.argv[2], 'card detector checksum mismatch: '+h; open('/app/models/card_detector.pt','wb').write(d); print('card detector:',len(d),'bytes',h)" "$CARD_DETECTOR_URL" "$CARD_DETECTOR_SHA256"

ENV NODE_ENV=production \
    STORAGE_DIR=/data \
    YOLO_MODEL=/app/models/card_detector.pt \
    YOLO_CONFIG_DIR=/tmp/ultralytics

EXPOSE 8080
CMD ["node", "server.js"]
