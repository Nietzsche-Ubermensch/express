const express = require('express');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const fs = require('fs');
const { execFile } = require('child_process');

const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;
const STORAGE = process.env.STORAGE_DIR || path.join(__dirname, 'data');
const UPLOAD_DIR = path.join(STORAGE, 'uploads');
const ENHANCED_DIR = path.join(STORAGE, 'enhanced');
const COMFYUI_URL = process.env.COMFYUI_URL || '';

[STORAGE, UPLOAD_DIR, ENHANCED_DIR].forEach(d => {
  if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
});

const app = express();
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));
app.use('/uploads', express.static(UPLOAD_DIR));
app.use('/enhanced', express.static(ENHANCED_DIR));

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, UPLOAD_DIR),
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname) || '.jpg';
    cb(null, uuidv4() + ext);
  },
});
const upload = multer({
  storage,
  limits: { fileSize: 25 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (file.mimetype.startsWith('image/')) cb(null, true);
    else cb(new Error('Only image files accepted'));
  },
});

const jobs = new Map();

app.get('/health', (_req, res) => res.status(200).send('ok'));

app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', uptime: process.uptime(), jobs: jobs.size, comfyui: COMFYUI_URL ? 'configured' : 'not configured', storage: STORAGE });
});

app.post('/api/cards/upload', upload.array('files', 50), async (req, res) => {
  if (!req.files || req.files.length === 0) return res.status(400).json({ error: 'No files uploaded' });
  const results = [];
  for (const file of req.files) {
    const cardId = path.basename(file.filename, path.extname(file.filename));
    const card = { id: cardId, fileName: file.originalname, filePath: file.path, url: `/uploads/${file.filename}`, enhancedUrl: null, metadata: {}, status: 'uploaded', createdAt: Date.now() };
    try { const meta = await runOCR(file.path); card.metadata = meta; card.status = 'ocr_complete'; } catch (e) { card.metadata = { error: e.message }; card.status = 'ocr_failed'; }
    jobs.set(cardId, card);
    results.push(card);
  }
  res.json({ cards: results.map(sanitize) });
});

app.post('/api/cards/:id/enhance', async (req, res) => {
  const card = jobs.get(req.params.id);
  if (!card) return res.status(404).json({ error: 'Card not found' });
  if (card.status === 'enhancing') return res.status(409).json({ error: 'Already processing' });
  card.status = 'enhancing';
  runEnhance(card).then(() => { card.status = 'enhanced'; }).catch((e) => { card.status = 'enhance_failed'; card.metadata.enhanceError = e.message; });
  res.json({ card: sanitize(card) });
});

app.get('/api/cards/:id', (req, res) => {
  const card = jobs.get(req.params.id);
  if (!card) return res.status(404).json({ error: 'Card not found' });
  res.json({ card: sanitize(card) });
});

app.get('/api/cards', (_req, res) => {
  const cards = Array.from(jobs.values()).sort((a, b) => b.createdAt - a.createdAt).map(sanitize);
  res.json({ cards });
});

app.post('/api/cards/enhance-all', async (_req, res) => {
  const pending = Array.from(jobs.values()).filter(c => c.status !== 'enhanced' && c.status !== 'enhancing');
  let started = 0;
  for (const card of pending) { card.status = 'enhancing'; runEnhance(card).then(() => { card.status = 'enhanced'; }).catch(() => { card.status = 'enhance_failed'; }); started++; }
  res.json({ started, total: jobs.size });
});

app.get('/api/cards/export/json', (_req, res) => {
  const data = Array.from(jobs.values()).map(c => ({ fileName: c.fileName, ...c.metadata, enhanced: c.status === 'enhanced', enhancedUrl: c.enhancedUrl }));
  res.setHeader('Content-Disposition', 'attachment; filename=cardcrop_export.json');
  res.json(data);
});

function runOCR(imagePath) {
  return new Promise((resolve, reject) => {
    const script = path.join(__dirname, 'ocr_worker.py');
    execFile('python3', [script, imagePath], { timeout: 30000 }, (err, stdout, stderr) => {
      if (err) return reject(new Error(stderr || err.message));
      try { resolve(JSON.parse(stdout)); } catch { reject(new Error('OCR output parse error')); }
    });
  });
}

function runEnhance(card) {
  return new Promise((resolve, reject) => {
    const script = path.join(__dirname, 'enhance_worker.py');
    const outName = card.id + '_enhanced.png';
    const outPath = path.join(ENHANCED_DIR, outName);
    execFile('python3', [script, card.filePath, outPath], { timeout: 60000 }, (err, stdout, stderr) => {
      if (err) return reject(new Error(stderr || err.message));
      card.enhancedUrl = `/enhanced/${outName}`;
      try { const result = JSON.parse(stdout); card.metadata.scratches_found = result.scratches_found; card.metadata.scratches_fixed = result.scratches_fixed; } catch {}
      resolve();
    });
  });
}

function sanitize(card) {
  return { id: card.id, fileName: card.fileName, url: card.url, enhancedUrl: card.enhancedUrl, metadata: card.metadata, status: card.status, createdAt: card.createdAt };
}

app.listen(PORT, '0.0.0.0', () => {
  console.log(`CardCrop AI running on http://0.0.0.0:${PORT}`);
  console.log(`  Storage: ${STORAGE}`);
  console.log(`  ComfyUI: ${COMFYUI_URL || 'not configured'}`);
});
