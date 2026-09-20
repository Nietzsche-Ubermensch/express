/**
 * VLM OCR — reads trading cards with a vision model instead of tesseract.
 *
 * Tesseract gets ~1 in 6 cards right on stylised wrestling card fonts and
 * invents plausible-looking garbage names ("Tung Hlingauet", "Bi Can Vas")
 * that can slip into a listing. A vision-language model reads them properly.
 *
 * Uses whichever key is present, in VLM_PROVIDER_ORDER (default below).
 * No key = {error:"no_vlm_configured"}; the tesseract path still works.
 */
const fs = require('fs');

const PROMPT = `Read this trading card image and return ONLY a JSON object, no other text, no markdown fences.

{
  "player_name": "the wrestler or athlete's name exactly as printed",
  "year": "4-digit year",
  "manufacturer": "Topps, Upper Deck, Panini, Leaf, etc",
  "set_name": "the product/set name",
  "promotion": "AEW, WWE, NXT, NBA, NFL, etc",
  "card_number": "card number if shown",
  "serial": "serial numbering like 136/199",
  "card_type": "base, patch, auto, relic, refractor, etc",
  "height": "listed height",
  "from_location": "listed hometown",
  "finishing_move": "listed finishing move",
  "bio_text": "the narrative paragraph if present",
  "is_back": true or false,
  "all_text": "every word you can read, in reading order"
}

Use null for anything not visible. Do not invent or guess any value.`;

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
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${process.env.OPENAI_API_KEY}` },
    body: JSON.stringify({
      model: model || process.env.OPENAI_VISION_MODEL || 'gpt-4o-mini',
      max_tokens: 900,
      messages: [{ role: 'user', content: [
        { type: 'text', text: PROMPT },
        { type: 'image_url', image_url: { url: `data:${mime};base64,${b64}` } },
      ]}],
    }),
  });
  if (!res.ok) throw new Error(`OpenAI ${res.status}: ${(await res.text()).slice(0, 300)}`);
  const d = await res.json();
  return parseJsonLoose(d.choices?.[0]?.message?.content);
}

async function readWithAnthropic(b64, mime, model) {
  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': process.env.ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: model || process.env.ANTHROPIC_VISION_MODEL || 'claude-sonnet-4-5',
      max_tokens: 900,
      messages: [{ role: 'user', content: [
        { type: 'image', source: { type: 'base64', media_type: mime, data: b64 } },
        { type: 'text', text: PROMPT },
      ]}],
    }),
  });
  if (!res.ok) throw new Error(`Anthropic ${res.status}: ${(await res.text()).slice(0, 300)}`);
  const d = await res.json();
  const txt = (d.content || []).filter(x => x.type === 'text').map(x => x.text).join('\n');
  return parseJsonLoose(txt);
}

async function readWithGoogle(b64, mime, model) {
  const key = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY;
  const m = model || process.env.GEMINI_VISION_MODEL || 'gemini-2.0-flash';
  const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${m}:generateContent?key=${key}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ parts: [ { text: PROMPT }, { inline_data: { mime_type: mime, data: b64 } } ] }],
      generationConfig: { temperature: 0, maxOutputTokens: 900 },
    }),
  });
  if (!res.ok) throw new Error(`Google ${res.status}: ${(await res.text()).slice(0, 300)}`);
  const d = await res.json();
  return parseJsonLoose(d.candidates?.[0]?.content?.parts?.[0]?.text);
}

async function readWithHF(b64, mime, model) {
  const m = model || process.env.HF_VISION_MODEL || 'zai-org/GLM-OCR';
  const res = await fetch('https://router.huggingface.co/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${process.env.HF_TOKEN}` },
    body: JSON.stringify({
      model: m, max_tokens: 900,
      messages: [{ role: 'user', content: [
        { type: 'text', text: PROMPT },
        { type: 'image_url', image_url: { url: `data:${mime};base64,${b64}` } },
      ]}],
    }),
  });
  if (!res.ok) throw new Error(`HF ${res.status}: ${(await res.text()).slice(0, 300)}`);
  const d = await res.json();
  return parseJsonLoose(d.choices?.[0]?.message?.content);
}

const READERS = {
  openai: readWithOpenAI,
  anthropic: readWithAnthropic,
  google: readWithGoogle,
  huggingface: readWithHF,
};

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
  const b64 = fs.readFileSync(filePath).toString('base64');
  const mime = filePath.toLowerCase().endsWith('.png') ? 'image/png' : 'image/jpeg';
  const t0 = Date.now();
  let meta;
  try {
    meta = await READERS[provider](b64, mime, opts.model);
  } catch (e) {
    return { error: String(e.message).slice(0, 400), vlm_provider: provider, vlm_ms: Date.now() - t0 };
  }
  meta.vlm_provider = provider;
  if (opts.model) meta.vlm_model = opts.model;
  meta.vlm_ms = Date.now() - t0;
  return meta;
}

// ── quality score that drives the 80% review gate ──
const WEIGHTS = { player_name: 30, year: 15, manufacturer: 15, set_name: 10, promotion: 10, card_type: 10, card_number: 5, serial: 5 };
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
    first.review_needed = first.quality < REVIEW_GATE;
    return first;
  }

  const [prov, model] = escalateTo.includes(':')
    ? escalateTo.split(':')
    : [first.vlm_provider, escalateTo];
  const second = await vlmRead(filePath, { ...opts, provider: prov, model });
  if (second.error) { first.review_needed = first.quality < REVIEW_GATE; return first; }
  second.quality = scoreQuality(second);

  const best = second.quality > first.quality ? second : first;
  best.vlm_escalated = true;
  best.vlm_first_pass = { provider: first.vlm_provider, quality: first.quality };
  best.review_needed = best.quality < REVIEW_GATE;
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

module.exports = { vlmRead, vlmReadGated, vlmReadMany, whichProvider, status, scoreQuality, REVIEW_GATE };
