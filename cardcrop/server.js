const express = require('express');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const fs = require('fs');
const { execFile } = require('child_process');
const vlm = require('./vlm_ocr');

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

// ── health ── Railway probes /health; the UI polls /api/health. Same payload.
app.get(['/health', '/api/health'], (_req, res) => {
  const s = vlm.status();
  res.json({
    status: 'ok',
    uptime: process.uptime(),
    jobs: jobs.size,
    comfyui: COMFYUI_URL ? 'configured' : 'not configured',
    vlm: s,
    review_gate: vlm.REVIEW_GATE,
    storage: STORAGE,
  });
});

// ── upload ──
app.post('/api/cards/upload', upload.array('files', 200), async (req, res) => {
  if (!req.files || req.files.length === 0) return res.status(400).json({ error: 'No files uploaded' });
  const results = [];
  for (const file of req.files) {
    const cardId = path.basename(file.filename, path.extname(file.filename));
    const card = {
      id: cardId, fileName: file.originalname, filePath: file.path,
      url: `/uploads/${file.filename}`, enhancedUrl: null,
      metadata: {}, status: 'queued', createdAt: Date.now(),
    };
    jobs.set(cardId, card);
    results.push(card);
  }
  res.json({ cards: results.map(sanitize) });

  // Read in the background so the upload returns immediately.
  for (const card of results) {
    card.status = 'processing';
    runOCR(card.filePath)
      .then(meta => { card.metadata = meta; card.status = 'ocr_complete'; })
      .catch(e => { card.metadata = { error: e.message }; card.status = 'failed'; });
  }
});

const optsFrom = q => ({
  provider: q.provider,
  model: q.model,
  escalate: q.escalate !== 'false',
});

// ── VLM: one card ──
app.post('/api/cards/:id/vlm', async (req, res) => {
  const card = jobs.get(req.params.id);
  if (!card) return res.status(404).json({ error: 'Card not found' });
  card.status = 'processing';
  try {
    const meta = await vlm.vlmReadGated(card.filePath, optsFrom(req.query));
    card.metadata = { ...card.metadata, ...meta };
    card.status = meta.error ? 'failed' : 'vlm_complete';
    res.json({ card: sanitize(card) });
  } catch (e) {
    card.status = 'failed';
    card.metadata = { ...card.metadata, error: e.message };
    res.status(500).json({ error: e.message });
  }
});

// ── VLM: whole batch, returns immediately and the UI polls ──
app.post('/api/cards/vlm-all', (req, res) => {
  let all = [...jobs.values()];
  if (req.query.only === 'pending') {
    all = all.filter(c => c.status !== 'vlm_complete' || c.metadata?.review_needed);
  }
  all.forEach(c => { c.status = 'processing'; });
  res.json({ started: all.length });

  vlm.vlmReadMany(all, optsFrom(req.query), (card, meta) => {
    card.metadata = { ...card.metadata, ...meta };
    card.status = meta.error ? 'failed' : 'vlm_complete';
  }).catch(() => {});
});

// ── enhance ──
app.post('/api/cards/:id/enhance', (req, res) => {
  const card = jobs.get(req.params.id);
  if (!card) return res.status(404).json({ error: 'Card not found' });
  if (card.status === 'enhancing') return res.status(409).json({ error: 'Already processing' });
  card.status = 'enhancing';
  runEnhance(card)
    .then(() => { card.status = 'enhanced'; })
    .catch(e => { card.status = 'failed'; card.metadata.enhanceError = e.message; });
  res.json({ card: sanitize(card) });
});

// ── read ──
app.get('/api/cards/:id', (req, res) => {
  const card = jobs.get(req.params.id);
  if (!card) return res.status(404).json({ error: 'Card not found' });
  res.json({ card: sanitize(card) });
});

app.get('/api/cards', (_req, res) => {
  const cards = [...jobs.values()].sort((a, b) => b.createdAt - a.createdAt).map(sanitize);
  res.json({ cards });
});

// ── delete ──
app.delete('/api/cards/:id', (req, res) => {
  const card = jobs.get(req.params.id);
  if (!card) return res.status(404).json({ error: 'Card not found' });
  try { fs.unlinkSync(card.filePath); } catch {}
  if (card.enhancedUrl) {
    try { fs.unlinkSync(path.join(ENHANCED_DIR, path.basename(card.enhancedUrl))); } catch {}
  }
  jobs.delete(req.params.id);
  res.json({ ok: true });
});

// ── export ──
app.get('/api/cards/export/json', (_req, res) => {
  const data = [...jobs.values()].map(c => ({
    fileName: c.fileName, status: c.status,
    ...c.metadata, enhancedUrl: c.enhancedUrl,
  }));
  res.setHeader('Content-Disposition', 'attachment; filename=cardcrop_export.json');
  res.json(data);
});

// ── python subprocesses ──
function runOCR(imagePath) {
  return new Promise((resolve, reject) => {
    const script = path.join(__dirname, 'ocr_worker.py');
    execFile('python3', [script, imagePath], { timeout: 30000 }, (err, stdout, stderr) => {
      if (err) return reject(new Error(stderr || err.message));
      try { resolve(JSON.parse(stdout)); }
      catch { reject(new Error('OCR output parse error')); }
    });
  });
}

function runEnhance(card) {
  return new Promise((resolve, reject) => {
    const script = path.join(__dirname, 'enhance_worker.py');
    const outName = card.id + '_enhanced.png';
    const outPath = path.join(ENHANCED_DIR, outName);
    execFile('python3', [script, card.filePath, outPath], { timeout: 120000 }, (err, stdout, stderr) => {
      if (err) return reject(new Error(stderr || err.message));
      card.enhancedUrl = `/enhanced/${outName}`;
      try {
        const r = JSON.parse(stdout);
        Object.assign(card.metadata, {
          scratches_found: r.scratches_found,
          rotation: r.rotation,
          output_size: r.output_size,
          orientation: r.orientation,
        });
      } catch {}
      resolve();
    });
  });
}

function sanitize(card) {
  return {
    id: card.id, fileName: card.fileName, url: card.url,
    enhancedUrl: card.enhancedUrl, metadata: card.metadata,
    status: card.status, createdAt: card.createdAt,
  };
}

// ── SPA fallback (Express 5 wildcard syntax) ──
app.get('/{*path}', (_req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
  const s = vlm.status();
  console.log(`CardCrop AI on http://0.0.0.0:${PORT}`);
  console.log(`  storage: ${STORAGE}`);
  console.log(`  vlm:     ${s.provider} (available: ${s.available.join(', ') || 'none'})`);
  console.log(`  comfyui: ${COMFYUI_URL || 'not configured'}`);
});
