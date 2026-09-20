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
const REGISTRY = path.join(STORAGE, 'jobs.json');
const COMFYUI_URL = process.env.COMFYUI_URL || '';
const YOLO_MODEL = process.env.YOLO_MODEL || path.join(__dirname, 'models', 'card_detector.pt');

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

// ══ persistent job registry ══
// Was in-memory only, so every Railway deploy replaced the container and
// orphaned the uploaded files on /data with no index pointing at them.
const jobs = new Map();
let saveTimer = null;

function snapshot() {
  return [...jobs.values()].map(c => ({
    id: c.id, fileName: c.fileName, filePath: c.filePath, url: c.url,
    enhancedUrl: c.enhancedUrl, metadata: c.metadata,
    status: (c.status === 'processing' || c.status === 'enhancing') ? 'queued' : c.status,
    createdAt: c.createdAt,
  }));
}

function persist() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try { fs.writeFileSync(REGISTRY, JSON.stringify(snapshot())); }
    catch (e) { console.warn('registry write failed:', e.message); }
  }, 400);
}

function restore() {
  let restored = 0;
  if (fs.existsSync(REGISTRY)) {
    try {
      for (const c of JSON.parse(fs.readFileSync(REGISTRY, 'utf8'))) {
        if (c.filePath && fs.existsSync(c.filePath)) { jobs.set(c.id, c); restored++; }
      }
    } catch (e) { console.warn('registry read failed:', e.message); }
  }
  let adopted = 0;
  try {
    for (const f of fs.readdirSync(UPLOAD_DIR)) {
      const id = path.basename(f, path.extname(f));
      if (jobs.has(id)) continue;
      const fp = path.join(UPLOAD_DIR, f);
      const enhName = id + '_enhanced.png';
      jobs.set(id, {
        id, fileName: f, filePath: fp, url: `/uploads/${f}`,
        enhancedUrl: fs.existsSync(path.join(ENHANCED_DIR, enhName)) ? `/enhanced/${enhName}` : null,
        metadata: {}, status: 'queued', createdAt: fs.statSync(fp).mtimeMs,
      });
      adopted++;
    }
  } catch (e) { console.warn('upload scan failed:', e.message); }
  if (restored || adopted) persist();
  return { restored, adopted };
}

// ══ health ══ Railway probes /health; the UI polls /api/health.
app.get(['/health', '/api/health'], (_req, res) => {
  res.json({
    status: 'ok',
    uptime: process.uptime(),
    jobs: jobs.size,
    comfyui: COMFYUI_URL ? 'configured' : 'not configured',
    vlm: vlm.status(),
    yolo: { available: fs.existsSync(YOLO_MODEL), path: YOLO_MODEL },
    review_gate: vlm.REVIEW_GATE,
    storage: STORAGE,
  });
});

// ══ upload ══
app.post('/api/cards/upload', upload.array('files', 200), (req, res) => {
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
  persist();
  res.json({ cards: results.map(sanitize) });

  for (const card of results) {
    card.status = 'processing';
    runOCR(card.filePath)
      .then(meta => { card.metadata = meta; card.status = 'ocr_complete'; })
      .catch(e => { card.metadata = { error: e.message }; card.status = 'failed'; })
      .finally(persist);
  }
});

const optsFrom = q => ({
  provider: q.provider, model: q.model, escalate: q.escalate !== 'false',
});
const enhanceOptsFrom = q => ({
  scale: q.scale ? Number(q.scale) : 2,
  descratch: q.descratch !== undefined ? Number(q.descratch) : 0.35,
  denoise: q.denoise !== undefined ? Number(q.denoise) : 0.2,
  sharpen: q.sharpen !== undefined ? Number(q.sharpen) : 0.5,
  contrast: q.contrast !== undefined ? Number(q.contrast) : 0.08,
  conservative: q.conservative !== 'false',
  autoRotate: q.autoRotate !== 'false',
});

// ══ VLM: one card ══
app.post('/api/cards/:id/vlm', async (req, res) => {
  const card = jobs.get(req.params.id);
  if (!card) return res.status(404).json({ error: 'Card not found' });
  card.status = 'processing';
  try {
    const meta = await vlm.vlmReadGated(card.filePath, optsFrom(req.query));
    card.metadata = { ...card.metadata, ...meta };
    card.status = meta.error ? 'failed' : 'vlm_complete';
    persist();
    res.json({ card: sanitize(card) });
  } catch (e) {
    card.status = 'failed';
    card.metadata = { ...card.metadata, error: e.message };
    persist();
    res.status(500).json({ error: e.message });
  }
});

// ══ VLM: batch, returns immediately and the UI polls ══
app.post('/api/cards/vlm-all', (req, res) => {
  let all = [...jobs.values()];
  if (req.query.only === 'pending') {
    all = all.filter(c => c.status !== 'vlm_complete' || c.metadata?.review_needed);
  }
  all.forEach(c => { c.status = 'processing'; });
  persist();
  res.json({ started: all.length });

  vlm.vlmReadMany(all, optsFrom(req.query), (card, meta) => {
    card.metadata = { ...card.metadata, ...meta };
    card.status = meta.error ? 'failed' : 'vlm_complete';
    persist();
  }).catch(() => {});
});

// ══ enhance ══
app.post('/api/cards/:id/enhance', (req, res) => {
  const card = jobs.get(req.params.id);
  if (!card) return res.status(404).json({ error: 'Card not found' });
  if (card.status === 'enhancing') return res.status(409).json({ error: 'Already processing' });
  card.status = 'enhancing';
  persist();
  runEnhance(card, enhanceOptsFrom(req.query))
    .then(() => { card.status = 'enhanced'; })
    .catch(e => { card.status = 'failed'; card.metadata.enhanceError = e.message; })
    .finally(persist);
  res.json({ card: sanitize(card) });
});

app.post('/api/cards/enhance-all', (req, res) => {
  const opts = enhanceOptsFrom(req.query);
  const all = [...jobs.values()].filter(c => c.status !== 'enhancing');
  all.forEach(c => { c.status = 'enhancing'; });
  persist();
  res.json({ started: all.length });

  (async () => {
    for (const card of all) {
      try { await runEnhance(card, opts); card.status = 'enhanced'; }
      catch (e) { card.status = 'failed'; card.metadata.enhanceError = e.message; }
      persist();
    }
  })();
});

// ══ read ══
app.get('/api/cards/:id', (req, res) => {
  const card = jobs.get(req.params.id);
  if (!card) return res.status(404).json({ error: 'Card not found' });
  res.json({ card: sanitize(card) });
});

app.get('/api/cards', (_req, res) => {
  const cards = [...jobs.values()].sort((a, b) => b.createdAt - a.createdAt).map(sanitize);
  res.json({ cards });
});

// ══ delete ══
app.delete('/api/cards/:id', (req, res) => {
  const card = jobs.get(req.params.id);
  if (!card) return res.status(404).json({ error: 'Card not found' });
  try { fs.unlinkSync(card.filePath); } catch {}
  if (card.enhancedUrl) {
    try { fs.unlinkSync(path.join(ENHANCED_DIR, path.basename(card.enhancedUrl))); } catch {}
  }
  jobs.delete(req.params.id);
  persist();
  res.json({ ok: true });
});

// ══ export ══
app.get('/api/cards/export/json', (_req, res) => {
  const data = [...jobs.values()].map(c => ({
    fileName: c.fileName, status: c.status, ...c.metadata, enhancedUrl: c.enhancedUrl,
  }));
  res.setHeader('Content-Disposition', 'attachment; filename=cardcrop_export.json');
  res.json(data);
});

// ══ python subprocesses ══
function runOCR(imagePath) {
  return new Promise((resolve, reject) => {
    execFile('python3', [path.join(__dirname, 'ocr_worker.py'), imagePath],
      { timeout: 30000 }, (err, stdout, stderr) => {
        if (err) return reject(new Error(stderr || err.message));
        try { resolve(JSON.parse(stdout)); }
        catch { reject(new Error('OCR output parse error')); }
      });
  });
}

function runEnhance(card, opts = {}) {
  return new Promise((resolve, reject) => {
    const outName = card.id + '_enhanced.png';
    const outPath = path.join(ENHANCED_DIR, outName);
    execFile('python3',
      [path.join(__dirname, 'enhance_worker.py'), card.filePath, outPath, JSON.stringify(opts)],
      { timeout: 180000 }, (err, stdout, stderr) => {
        if (err) return reject(new Error(stderr || err.message));
        card.enhancedUrl = `/enhanced/${outName}`;
        try {
          const r = JSON.parse(stdout);
          Object.assign(card.metadata, {
            scratches_found: r.scratches_found, rotation: r.rotation,
            output_size: r.output_size, orientation: r.orientation,
            yolo_conf: r.yolo_conf, yolo_cropped: r.yolo_cropped,
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

// ══ SPA fallback (Express 5 wildcard syntax) ══
app.get('/{*path}', (_req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const boot = restore();

app.listen(PORT, '0.0.0.0', () => {
  const s = vlm.status();
  console.log(`CardCrop AI on http://0.0.0.0:${PORT}`);
  console.log(`  storage: ${STORAGE}`);
  console.log(`  jobs:    ${jobs.size} (${boot.restored} from registry, ${boot.adopted} adopted from disk)`);
  console.log(`  yolo:    ${fs.existsSync(YOLO_MODEL) ? YOLO_MODEL : 'no weights — whole-frame fallback'}`);
  console.log(`  vlm:     ${s.provider} (available: ${s.available.join(', ') || 'none'})`);
  console.log(`  comfyui: ${COMFYUI_URL || 'not configured'}`);
});

for (const sig of ['SIGTERM', 'SIGINT']) {
  process.on(sig, () => {
    clearTimeout(saveTimer);
    try { fs.writeFileSync(REGISTRY, JSON.stringify(snapshot())); } catch {}
    process.exit(0);
  });
}
