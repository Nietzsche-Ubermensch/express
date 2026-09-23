/** Wrestling-card identification. Provider output always requires human review.
 * The field-completeness score is not a calibrated accuracy/confidence score.
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');

const PROMPT = `You are a professional wrestling trading card cataloger. This app accepts only professional WRESTLING trading cards (WWE, AEW, NXT, WCW, TNA/Impact, ROH, NJPW, or another pro wrestling promotion).

The operator only uploads professional wrestling cards. Never refuse or return an error because of category. Set "category" to "wrestling" unless the card shows clear evidence of another sport (for example an NBA, NFL, MLB or NHL logo or team), in which case set "category" to "other_sport" and still read every field. Cards may be small on a large background, die-cut, sideways or upside down; read them anyway.

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

If the text is sideways or upside down, say so in text_top and use null for any name you cannot actually read. Never identify a wrestler from their appearance — only from printed text.

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
// OpenAI's "try again in 469ms" is useless when the org's whole TPM bucket is
// full (seen live: Limit 200000, Used 200000). The hint may only lengthen the
// exponential floor, never shorten it. Default budget ~60s spans a TPM window.
function retryDelay(res, body, attempt) {
  const floor = Math.min(2000 * 2 ** attempt, 30000);
  let hint = 0;
  const ra = Number(res.headers.get('retry-after'));
  if (Number.isFinite(ra) && ra > 0) hint = ra * 1000;
  const m = /try again in ([\d.]+)\s*(ms|s)/i.exec(body || '');
  if (m) hint = Math.max(hint, Number(m[1]) * (m[2].toLowerCase() === 'ms' ? 1 : 1000));
  return Math.min(Math.max(floor, hint + 250), 60000);
}
/** POST with retry on 429 / 5xx. Throws `${label} ${status}: body` on final failure. */
async function postJSON(url, init, label, attempts = Number(process.env.VLM_RETRIES) || 6) {
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

async function readWithOpenAI(b64, mime, model, prompt = PROMPT) {
  const d = await postJSON('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${process.env.OPENAI_API_KEY}` },
    body: JSON.stringify({
      model: model || process.env.OPENAI_VISION_MODEL || 'gpt-4o-mini',
      max_tokens: 1200,
      messages: [{ role: 'user', content: [
        { type: 'text', text: prompt },
        { type: 'image_url', image_url: { url: `data:${mime};base64,${b64}` } },
      ]}],
    }),
  }, 'OpenAI');
  return parseJsonLoose(d.choices?.[0]?.message?.content);
}

async function readWithAnthropic(b64, mime, model, prompt = PROMPT) {
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
        { type: 'text', text: prompt },
      ]}],
    }),
  }, 'Anthropic');
  const txt = (d.content || []).filter(x => x.type === 'text').map(x => x.text).join('\n');
  return parseJsonLoose(txt);
}

async function readWithGoogle(b64, mime, model, prompt = PROMPT) {
  const key = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY;
  const m = model || process.env.GEMINI_VISION_MODEL || 'gemini-3.6-flash';
  const d = await postJSON(`https://generativelanguage.googleapis.com/v1beta/models/${m}:generateContent?key=${key}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ parts: [ { text: prompt }, { inline_data: { mime_type: mime, data: b64 } } ] }],
      // maxOutputTokens includes hidden thinking tokens on Gemini 3.x; with
      // default (medium) thinking, a 1200 cap was spent reasoning and the JSON
      // came back truncated. Card reading needs no deliberation.
      generationConfig: {
        temperature: 0,
        maxOutputTokens: Number(process.env.GEMINI_MAX_OUTPUT_TOKENS) || 8192,
        responseMimeType: 'application/json',
        thinkingConfig: { thinkingLevel: process.env.GEMINI_THINKING_LEVEL || 'minimal' },
      },
    }),
  }, 'Google');
  const cand = d.candidates?.[0];
  const text = (cand?.content?.parts || []).filter(x => !x.thought).map(x => x.text || '').join('');
  if (!text && cand?.finishReason) throw new Error(`Google finishReason=${cand.finishReason} with no text (thoughts=${d.usageMetadata?.thoughtsTokenCount ?? '?'})`);
  const out = parseJsonLoose(text);
  if (out.parse_error && cand?.finishReason === 'MAX_TOKENS') throw new Error('Google output truncated at maxOutputTokens (raise GEMINI_MAX_OUTPUT_TOKENS or lower GEMINI_THINKING_LEVEL)');
  return out;
}

async function readWithHF(b64, mime, model, prompt = PROMPT) {
  const m = model || process.env.HF_VISION_MODEL || 'zai-org/GLM-OCR';
  const d = await postJSON('https://router.huggingface.co/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${process.env.HF_TOKEN}` },
    body: JSON.stringify({
      model: m, max_tokens: 1200,
      messages: [{ role: 'user', content: [
        { type: 'text', text: prompt },
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
// Asking the model which way text points ("left"/"right") proved unreliable
// on real scans: card 0996 was reported "down" then "right" and ended up sent
// upside-down. Models judge *comparatively* far better, so when a read looks
// wrong we show all four rotations as numbered panels and ask which is upright.
const PANEL_ROTATION = [0, 90, 180, 270]; // degrees clockwise, panels 1..4
const PICK_PROMPT = `This image shows ONE trading card four times, in panels numbered 1, 2, 3, 4 (the number is above each panel). Each panel is rotated differently. Which single panel shows the card's printed text upright — readable left-to-right, not sideways, not upside down? Return ONLY JSON: {"upright_panel": 1, 2, 3 or 4}`;

function tmpPath(ext) {
  return path.join(os.tmpdir(), `vlm_${process.pid}_${Date.now()}_${Math.random().toString(36).slice(2)}.${ext}`);
}

function makePanels(src) {
  const out = tmpPath('jpg');
  execFileSync('python3', ['-c', `import cv2,sys,numpy as np
im=cv2.imread(sys.argv[1]); assert im is not None
rots=[im,cv2.rotate(im,cv2.ROTATE_90_CLOCKWISE),cv2.rotate(im,cv2.ROTATE_180),cv2.rotate(im,cv2.ROTATE_90_COUNTERCLOCKWISE)]
S=512; tiles=[]
for i,r in enumerate(rots):
    h,w=r.shape[:2]; k=S/max(h,w); r=cv2.resize(r,(max(1,round(w*k)),max(1,round(h*k))),interpolation=cv2.INTER_AREA)
    t=np.full((S+70,S+20,3),255,np.uint8); y=70+(S-r.shape[0])//2; x=10+(S-r.shape[1])//2
    t[y:y+r.shape[0],x:x+r.shape[1]]=r
    cv2.putText(t,str(i+1),((S+20)//2-18,55),cv2.FONT_HERSHEY_SIMPLEX,1.8,(0,0,210),5)
    tiles.append(t)
cv2.imwrite(sys.argv[2],np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])]),[cv2.IMWRITE_JPEG_QUALITY,90])`, src, out], { timeout: 30000 });
  return out;
}

function rotateImage(src, degreesCW) {
  const out = tmpPath('jpg');
  const code = { 90: 'ROTATE_90_CLOCKWISE', 180: 'ROTATE_180', 270: 'ROTATE_90_COUNTERCLOCKWISE' }[degreesCW];
  execFileSync('python3', ['-c',
    'import cv2,sys\nim=cv2.imread(sys.argv[1])\nassert im is not None\n' +
    `cv2.imwrite(sys.argv[2], cv2.rotate(im, cv2.${code}), [cv2.IMWRITE_JPEG_QUALITY, 95])`,
    src, out], { timeout: 30000 });
  return out;
}

// ── grounding ──
// A name is only kept if the model's own transcription contains it.
const norm = t => String(t || '').replace(/[\u2122\u00ae\u00a9\u2120]/g, ' ').normalize('NFD')
  .replace(/[\u0300-\u036f]/g, '').toUpperCase().replace(/[^A-Z0-9]+/g, ' ').trim();
function groundName(meta) {
  if (!meta) return meta;
  const keep = (Array.isArray(meta.warnings) ? meta.warnings : []).filter(w => w !== 'name_not_in_read_text');
  if (!meta.player_name) { meta.name_grounded = null; meta.player_name_unverified = null; meta.warnings = keep; return meta; }
  const text = ` ${norm(meta.all_text)} `;
  const tokens = norm(meta.player_name).split(' ').filter(t => t.length >= 3);
  const grounded = tokens.length > 0 && tokens.every(t => text.includes(` ${t} `));
  meta.name_grounded = grounded;
  meta.player_name_unverified = grounded ? null : meta.player_name;
  meta.warnings = grounded ? keep : [...keep, 'name_not_in_read_text'];
  if (!grounded) meta.player_name = null;
  return meta;
}

async function readOnce(provider, filePath, model, prompt) {
  const b64 = fs.readFileSync(filePath).toString('base64');
  const mime = filePath.toLowerCase().endsWith('.png') ? 'image/png' : 'image/jpeg';
  return READERS[provider](b64, mime, model, prompt);
}

/** Does this read suggest the image was not upright? */
function looksMisoriented(meta) {
  if (!meta || meta.parse_error) return false;
  if (meta.error) return CATEGORY_ERRORS.has(meta.error);
  const top = String(meta.text_top || 'up').toLowerCase();
  if (top !== 'up') return true;
  if (!meta.player_name) return true;
  return groundName({ player_name: meta.player_name, all_text: meta.all_text }).name_grounded !== true;
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
  let meta, applied = 0, method = 'original', tmp = null, pickedPanel = null;
  try {
    meta = await readOnce(provider, filePath, opts.model);
    if (opts.orient !== false && looksMisoriented(meta)) {
      tmp = makePanels(filePath);
      const pick = await readOnce(provider, tmp, opts.model, PICK_PROMPT);
      fs.unlinkSync(tmp); tmp = null;
      pickedPanel = Number(pick && pick.upright_panel);
      const deg = PANEL_ROTATION[pickedPanel - 1];
      method = 'panel_pick';
      if (deg) {
        tmp = rotateImage(filePath, deg);
        const reread = await readOnce(provider, tmp, opts.model);
        if (reread && !reread.parse_error) { meta = reread; applied = deg; }
      }
    }
  } catch (e) {
    return { error: String(e.message).slice(0, 400), vlm_provider: provider, vlm_ms: Date.now() - t0 };
  } finally {
    if (tmp) { try { fs.unlinkSync(tmp); } catch {} }
  }
  meta = validateWrestling(meta);
  if (!meta.error) groundName(meta);
  meta.read_rotation = applied;
  meta.orientation_method = method;
  meta.orientation_panel = pickedPanel;
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

// Category doubt is a review flag, not a failure: the operator uploads only
// wrestling cards, and hard rejection threw away 29/428 real wrestling cards
// (die-cuts on large backgrounds, upside-down scans) in the first full batch.
const CATEGORY_ERRORS = new Set(['unsupported_category', 'category_unverified']);
const OTHER_SPORT = /^(NBA|NFL|MLB|NHL|UFC|MLS|baseball|basketball|football|hockey|soccer|MMA)$/i;
function validateWrestling(meta) {
  if (!meta || typeof meta !== 'object' || Array.isArray(meta) || meta.parse_error) {
    return { error: 'invalid_identification_response', review_needed: true };
  }
  if (meta.error && !CATEGORY_ERRORS.has(meta.error)) return { ...meta, review_needed: true };
  const out = { ...meta, review_needed: true };
  const warnings = [];
  if (CATEGORY_ERRORS.has(meta.error)) { delete out.error; warnings.push('category_' + meta.error); }
  if (out.category !== 'wrestling' || OTHER_SPORT.test(out.promotion || '')) {
    warnings.push('category_not_confirmed_wrestling');
  }
  out.warnings = [...(Array.isArray(meta.warnings) ? meta.warnings : []), ...warnings];
  return out;
}

module.exports = { groundName, retryDelay, looksMisoriented, PANEL_ROTATION, validateWrestling, vlmRead, vlmReadGated, vlmReadMany, whichProvider, status, scoreQuality, REVIEW_GATE };
