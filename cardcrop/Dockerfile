FROM node:20-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3 /usr/bin/python
WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev
RUN pip3 install --no-cache-dir --break-system-packages opencv-python-headless numpy Pillow pytesseract
COPY . .
RUN mkdir -p /data/uploads /data/enhanced
ENV NODE_ENV=production STORAGE_DIR=/data
EXPOSE 8080
CMD ["node", "server.js"]
