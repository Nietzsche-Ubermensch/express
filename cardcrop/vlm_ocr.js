/** Wrestling-card identification. Provider output always requires human review.
 * The field-completeness score is not a calibrated accuracy/confidence score.
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');

const PROMPT = `You are a professional wrestling trading card cataloger. This app accepts only professional WRESTLING trading cards (WWE, AEW, NXT, WCW, TNA/Impact, ROH, NJPW, or another pro wrestling promotion).

First inspect the image. For a clearly non-wrestling card return {"error":"unsupported_category"}. For an uncertain category return {"error":"category_unverified"}. Never relabel another sport as wrestling.

Read every piece of text on the card. Wrestling cards often use metallic foil, holographic refractor patterns, or heavily stylized fonts — look carefully through any visual noise for the actual printed text. Serial numbering is often stamped in silver, gold, or white ink and may appear on any edge or corner.

Return ONLY a JSON object. No other text, no markdown fences.

{
  "category": "wrestling",
  "player_name": "wrestler's ring name exactly as printed (e.g. 'The Rock', 'Stone Cold Steve Austin', 'Kenny Omega')",
  "real_name": "shoot name if printed separately from ring name, else null",
  "year": "4-digit year from the card or copyright line",
  "manufacturer": "Topps, Upper Deck, Panini, Fleer, Pacific, Comic Images, Duocards, etc",
  "set_name": "product/set name (e.g. 'Heritage', 'Transcendent', 'Prizm', 'Chrome', 'Undisputed')",
  "subset": "insert set or subset name if this is not a base card (e.g. 'Autographs', 'Hall of Fame', 'Legendary Cuts')",
  "promotion": "WWE, AEW, NXT, WCW, ECW, TNA, Impact, ROH, NJPW, or whichever promotion appears",
  "card_number": "card number exactly as printed (e.g. '42', 'HF-12', 'NXT-7')",
  "serial": "serial numbering if stamped (e.g. '136/199', '023/050'). Read carefully — often stamped faintly",
  "card_type": "base, autograph, auto, patch, relic, memorabilia, refractor, prizm, printing plate, kiss, 1/1, etc",
  "parallel": "parallel variant name if any (e.g. 'Gold', 'Red', 'Superfractor', 'Shimmer', 'Black', 'Camo')",
  "tag_team": "tag team name if listed (e.g. 'The Hardy Boyz', 'The New Day')",
  "stable": "faction/stable if listed (e.g. 'nWo', 'D-Generation X', 'The Shield')",
  "weight_class": "billed weight if shown",
  "height": "billed height if shown",
  "from_location": "billed hometown if shown (e.g. 'Parts Unknown', 'Venice Beach, CA')",
  "finishing_move": "signature/finishing move if listed",
  "championship": "championship title shown or mentioned on card",
  "era": "era if identifiable from card design or text (e.g. 'Attitude Era', 'Ruthless Aggression')",
  "bio_text": "the narrative/biographical paragraph if present on a card back",
  "text_top": "which direction the TOP of the main printed letters points in this image: up, left, right, or down",
  "is_back": true if this is the back side of the card, false if front,
  "all_text": "every word you can read on the card, in reading order, including fine print and copyright"
}

If the text is sideways or upside down, still report text_top accurately and use null for any name you cannot actually read. Never identify a wrestler from their appearance — only from printed text.

Use null for anything not visible. Do not invent or guess any value — if text is unreadable through foil or glare, use null rather than guessing.`;

const PROVIDERS = {
  openai: () => process.env.OPENAI_API_KEY,
  anthropic: () => process.env.ANTHROPIC_API_KEY,
  google: () => process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY,
  huggingface: () => process.env.HF_TOKEN,
};

function order() {
  const raw = process.env.VLM_PROVIDER_ORDER || 'openai,anthropic,google,huggingface';
  return raw.split(',').map(s => s.trim()).filter(s => PROVIDERS[s]);
}

function whichProvider() {
  for (const p of order()) if (PROVIDERS[p]()) return p;
  return null;
}

function status() {
  const available = order().filter(p => PROVIDERS[p]());
  return {
    provider: available[0] || 'not configured',
    available,
    checked: order(),
  };
}

const sleep = ms => new Promise(r => setTimeout(r, ms));
function retryDelay(res, body, attempt) {
  const ra = Number(res.headers.get('retry-after'));
  if (Number.isFinite(ra) && ra > 0) return Math.min(ra * 1000, 60000);
  const m = /try again in ([\d.]+)\s*(ms|s)/i.exec(body || '');
  if (m) return Math.min(Number(m[1]) * (m[2].toLowerCase() === 'ms' ? 1 : 1000) + 250, 60000);
  return Math.min(2000 * 2 ** attempt, 30000);
}
/** POST with retry on 429 / 5xx. Throws `${label} ${status}: body` on final failure. */
async function postJSON(url, init, label, attempts = Number(process.env.VLM_RETRIES) || 4) {
  for (let i = 0; ; i++) {
    const res = await fetch(url, init);
    if (res.ok) return res.json();
    const body = await res.text();
    if ((res.status === 429 || res.status >= 500) && i < attempts - 1) {
      await sleep(retryDelay(res, body, i));
      continue;
    }
    throw new Error(`${label} ${res.status}: ${body.slice(0, 300)}`);
  }
}

function parseJsonLoose(raw) {
  let c = (raw || '').trim();
  if (c.includes('```')) {
    for (let part of c.split('```')) {
      part = part.trim();
      if (part.toLowerCase().startsWith('json')) part = part.slice(4).trim();
      if (part.startsWith('{')) { c = part; break; }
    }
  }
  const a = c.indexOf('{'), b = c.lastIndexOf('}');
  if (a !== -1 && b > a) c = c.slice(a, b + 1);
  try { return JSON.parse(c); }
  catch { return { parse_error: true, raw_output: String(raw).slice(0, 600) }; }
}

async function readWithOpenAI(b64, mime, model) {
  const d = await postJSON('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${process.env.OPENAI_API_KEY}` },
    body: JSON.stringify({
      model: model || process.env.OPENAI_VISION_MODEL || 'gpt-4o-mini',
      max_tokens: 1200,
      messages: [{ role: 'user', content: [
        { type: 'text', text: PROMPT },
        { type: 'image_url', image_url: { url: `data:${mime};base64,${b64}` } },
      ]}],
    }),
  }, 'OpenAI');
  return parseJsonLoose(d.choices?.[0]?.message?.content);
}

async function readWithAnthropic(b64, mime, model) {
  const d = await postJSON('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': process.env.ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: model || process.env.ANTHROPIC_VISION_MODEL || 'claude-sonnet-4-5',
      max_tokens: 1200,
      messages: [{ role: 'user', content: [
        { type: 'image', source: { type: 'base64', media_type: mime, data: b64 } },
        { type: 'text', text: PROMPT },
      ]}],
    }),
  }, 'Anthropic');
  const txt = (d.content || []).filter(x => x.type === 'text').map(x => x.text).join('\n');
  return parseJsonLoose(txt);
}

async function readWithGoogle(b64, mime, model) {
  const key = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY;
  const m = model || process.env.GEMINI_VISION_MODEL || 'gemini-3.6-flash';
  const d = await postJSON(`https://generativelanguage.googleapis.com/v1beta/models/${m}:generateContent?key=${key}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ parts: [ { text: PROMPT }, { inline_data: { mime_type: mime, data: b64 } } ] }],
      generationConfig: { temperature: 0, maxOutputTokens: 1200 },
    }),
  }, 'Google');
  return parseJsonLoose(d.candidates?.[0]?.content?.parts?.[0]?.text);
}

async function readWithHF(b64, mime, model) {
  const m = model || process.env.HF_VISION_MODEL || 'zai-org/GLM-OCR';
  const d = await postJSON('https://router.huggingface.co/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${process.env.HF_TOKEN}` },
    body: JSON.stringify({
      model: m, max_tokens: 1200,
      messages: [{ role: 'user', content: [
        { type: 'text', text: PROMPT },
        { type: 'image_url', image_url: { url: `data:${mime};base64,${b64}` } },
      ]}],
    }),
  }, 'HF');
  return parseJsonLoose(d.choices?.[0]?.message?.content);
}

const READERS = {
  openai: readWithOpenAI,
  anthropic: readWithAnthropic,
  google: readWithGoogle,
  huggingface: readWithHF,
};

// ── orientation ──
// Degrees CLOCKWISE to apply so text reads upright, keyed by where the tops
// of the letters currently point. Sideways scans were the main cause of
// invented names: the model could not read the nameplate and guessed.
const TURN = { up: 0, left: 90, down: 180, right: 270 };
const MAX_TURNS = Number(process.env.VLM_MAX_TURNS ?? 2);

function rotateImage(src, degreesCW) {
  const out = path.join(os.tmpdir(), `vlm_${process.pid}_${Date.now()}_${Math.random().toString(36).slice(2)}.jpg`);
  const code = { 90: 'ROTATE_90_CLOCKWISE', 180: 'ROTATE_180', 270: 'ROTATE_90_COUNTERCLOCKWISE' }[degreesCW];
  execFileSync('python3', ['-c',
    'import cv2,sys\nim=cv2.imread(sys.argv[1])\nassert im is not None\n' +
    `cv2.imwrite(sys.argv[2], cv2.rotate(im, cv2.${code}), [cv2.IMWRITE_JPEG_QUALITY, 95])`,
    src, out], { timeout: 30000 });
  return out;
}

// ── grounding ──
// A name is only kept if the model's own transcription contains it.
const norm = t => String(t || '').toUpperCase().normalize('NFKD').replace(/[^A-Z0-9]+/g, ' ').trim();
function groundName(meta) {
  if (!meta) return meta;
  if (!meta.player_name) { meta.name_grounded = null; meta.player_name_unverified = null; meta.warnings = []; return meta; }
  const text = ` ${norm(meta.all_text)} `;
  const tokens = norm(meta.player_name).split(' ').filter(t => t.length >= 3);
  const grounded = tokens.length > 0 && tokens.every(t => text.includes(` ${t} `));
  meta.name_grounded = grounded;
  meta.player_name_unverified = grounded ? null : meta.player_name;
  meta.warnings = grounded ? [] : ['name_not_in_read_text'];
  if (!grounded) meta.player_name = null;
  return meta;
}

async function readOnce(provider, filePath, model) {
  const b64 = fs.readFileSync(filePath).toString('base64');
  const mime = filePath.toLowerCase().endsWith('.png') ? 'image/png' : 'image/jpeg';
  return READERS[provider](b64, mime, model);
}

/** Read one card image with whichever VLM is configured. */
async function vlmRead(filePath, opts = {}) {
  const provider = opts.provider && PROVIDERS[opts.provider] && PROVIDERS[opts.provider]()
    ? opts.provider
    : whichProvider();
  if (!provider) {
    return {
      error: 'no_vlm_configured',
      hint: 'Set OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY or HF_TOKEN in Railway > Variables, then redeploy.',
    };
  }
  const t0 = Date.now();
  let meta, applied = 0, turns = 0, tmp = null;
  try {
    meta = await readOnce(provider, filePath, opts.model);
    // Re-read upright. Measured against the original so errors don't compound;
    // a wrong left/right guess shows up as "down" and gets one more correction.
    while (opts.orient !== false && turns < MAX_TURNS && meta && !meta.parse_error) {
      const step = TURN[String(meta.text_top || 'up').toLowerCase()];
      if (!step) break;
      applied = (applied + step) % 360;
      turns++;
      if (tmp) { try { fs.unlinkSync(tmp); } catch {} tmp = null; }
      if (applied === 0) { meta = await readOnce(provider, filePath, opts.model); continue; }
      tmp = rotateImage(filePath, applied);
      meta = await readOnce(provider, tmp, opts.model);
    }
  } catch (e) {
    return { error: String(e.message).slice(0, 400), vlm_provider: provider, vlm_ms: Date.now() - t0 };
  } finally {
    if (tmp) { try { fs.unlinkSync(tmp); } catch {} }
  }
  meta = validateWrestling(meta);
  if (!meta.error) groundName(meta);
  meta.read_rotation = applied;
  meta.orientation_passes = turns;
  meta.vlm_provider = provider;
  if (opts.model) meta.vlm_model = opts.model;
  meta.vlm_ms = Date.now() - t0;
  return meta;
}

// ── quality score that drives the 80% review gate ──
const WEIGHTS = {
  player_name: 25, year: 12, manufacturer: 12, set_name: 10,
  promotion: 10, card_type: 8, card_number: 5, serial: 5,
  parallel: 5, subset: 5, finishing_move: 3,
};
const REVIEW_GATE = Number(process.env.VLM_REVIEW_GATE) || 80;
function scoreQuality(m) {
  return Object.entries(WEIGHTS).reduce((s, [k, w]) => s + (m && m[k] ? w : 0), 0);
}

/** Read, score, and escalate to a stronger model when under the gate. */
async function vlmReadGated(filePath, opts = {}) {
  const first = await vlmRead(filePath, opts);
  if (first.error) return first;
  first.quality = scoreQuality(first);

  const escalateTo = process.env.VLM_ESCALATE_MODEL;
  if (first.quality >= REVIEW_GATE || opts.escalate === false || !escalateTo) {
    first.review_needed = true;
    return first;
  }

  const [prov, model] = escalateTo.includes(':')
    ? escalateTo.split(':')
    : [first.vlm_provider, escalateTo];
  const second = await vlmRead(filePath, { ...opts, provider: prov, model });
  if (second.error) { first.review_needed = true; return first; }
  second.quality = scoreQuality(second);

  const best = second.quality > first.quality ? second : first;
  best.vlm_escalated = true;
  best.vlm_first_pass = { provider: first.vlm_provider, quality: first.quality };
  best.review_needed = true;
  return best;
}

/** Bounded-concurrency batch read. */
async function vlmReadMany(cards, opts = {}, onDone = () => {}) {
  const limit = Number(process.env.VLM_CONCURRENCY) || 3;
  const queue = [...cards];
  const workers = Array.from({ length: Math.min(limit, queue.length) }, async () => {
    while (queue.length) {
      const card = queue.shift();
      try {
        const meta = await vlmReadGated(card.filePath, opts);
        onDone(card, meta);
      } catch (e) {
        onDone(card, { error: String(e.message).slice(0, 400) });
      }
    }
  });
  await Promise.all(workers);
}

function validateWrestling(meta) {
  if (!meta || typeof meta !== 'object' || Array.isArray(meta) || meta.parse_error) {
    return { error: 'invalid_identification_response', review_needed: true };
  }
  if (meta.error) return { ...meta, review_needed: true };
  if (meta.category !== 'wrestling' || /^(NBA|NFL|MLB|NHL|UFC|baseball|basketball|football|hockey|MMA)$/i.test(meta.promotion || '')) {
    return { error: 'unsupported_or_unverified_category', review_needed: true };
  }
  return { ...meta, review_needed: true };
}

module.exports = { groundName, retryDelay, TURN, validateWrestling, vlmRead, vlmReadGated, vlmReadMany, whichProvider, status, scoreQuality, REVIEW_GATE };
