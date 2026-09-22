/** Wrestling-card identification. Provider output always requires human review.
 * The field-completeness score is not a calibrated accuracy/confidence score.
 */
const fs = require('fs');

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
  "is_back": true if this is the back side of the card, false if front,
  "all_text": "every word you can read on the card, in reading order, including fine print and copyright"
}

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
      max_tokens: 1200,
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
      max_tokens: 1200,
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
  const m = model || process.env.GEMINI_VISION_MODEL || 'gemini-3.6-flash';
  const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${m}:generateContent?key=${key}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ parts: [ { text: PROMPT }, { inline_data: { mime_type: mime, data: b64 } } ] }],
      generationConfig: { temperature: 0, maxOutputTokens: 1200 },
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
      model: m, max_tokens: 1200,
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
  meta = validateWrestling(meta);
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

module.exports = { validateWrestling, vlmRead, vlmReadGated, vlmReadMany, whichProvider, status, scoreQuality, REVIEW_GATE };
