/**
 * PolicyDesk - client-side RAG engine (runs entirely in the browser).
 *
 * The query is embedded locally with Transformers.js (same BGE model used to
 * embed the passages at build time). Retrieval = dense cosine + BM25, fused with
 * Reciprocal Rank Fusion, diversified with MMR, gated by a calibrated abstention
 * threshold. Answers are extractive (top cited passages) - no server, no API key.
 *
 * This is a faithful JS port of the Python engine in ../src/.
 */
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6';
import * as pdfjsLib from 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4/build/pdf.min.mjs';
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4/build/pdf.worker.min.mjs';

env.allowLocalModels = false;          // fetch the model from the HF CDN
env.useBrowserCache = true;            // cache it after first load

const QUERY_PREFIX = 'Represent this sentence for searching relevant passages: ';
// Calibrated for the BFSI q8 + CLS-pooled BGE embeddings (build_embeddings.mjs):
// in-corpus min≈0.721, out-of-corpus max≈0.470 → 0.60 sits in a wide clean gap.
const ABSTAIN_THRESHOLD = 0.60;
const DIM = 384;
const REFUSAL = "I couldn't find that in the policy documents.";

// In-browser writer model (open, runs locally via WebGPU/WASM; no key, no server).
// 0.5B keeps the one-time download light (~0.4GB) and generation fast.
const GEN_MODEL = 'onnx-community/Qwen2.5-0.5B-Instruct';
const GEN_SYSTEM = `You are a warm, helpful policy advisor at a bank/insurance company, \
speaking to a customer. Answer in plain, friendly, natural language.
- Lead with the direct answer, then briefly explain it. Keep it to 2-4 sentences.
- Always include the exact figures that matter (amounts, percentages, waiting periods, timelines).
- Cite the excerpt number(s) you used, like [1].
- Use ONLY the information in the excerpts. Never guess or use outside knowledge. If the \
excerpts do not answer it, say so politely.`;

// In-browser speech-to-text (Whisper via Transformers.js; English, small + fast).
const ASR_MODEL = 'Xenova/whisper-tiny.en';

let MAT, CHUNKS, N, BM25, extractor, META;
let LLM_ENABLED = false;    // hosted GLM endpoint (probed at boot)
let VOICE_ENABLED = false;  // hosted Groq Whisper endpoint (probed at boot)
let WRITTEN_MODE = false;   // in-browser written answers (user toggles on)

// ---------------------------------------------------------------------------
// Load static index
// ---------------------------------------------------------------------------
async function loadIndex() {
  const [meta, chunks, embBuf] = await Promise.all([
    fetch('data/meta.json').then(r => r.json()),
    fetch('data/chunks.json').then(r => r.json()),
    fetch('data/embeddings.bin').then(r => r.arrayBuffer()),
  ]);
  META = meta;
  CHUNKS = chunks;
  N = chunks.length;
  MAT = new Float32Array(embBuf);       // N*DIM, row-major, L2-normalised
  BM25 = new BM25Index(chunks.map(c => c.text));
  const docCount = new Set(chunks.map(c => c.doc_id)).size;
  document.getElementById('b-docs').textContent = `${docCount} policy documents`;
}

// ---------------------------------------------------------------------------
// BM25 (Okapi) - ported from src/bm25.py
// ---------------------------------------------------------------------------
const STOP = new Set(("the a an and or of to in is are for on with as by that this be we " +
  "it at from can").split(' '));
function tokenize(text) {
  const out = [];
  for (const m of text.toLowerCase().matchAll(/[a-z0-9]+/g)) {
    const t = m[0];
    if (t.length > 1 && !STOP.has(t)) out.push(t);
  }
  return out;
}
class BM25Index {
  constructor(corpus, k1 = 1.5, b = 0.75) {
    this.k1 = k1; this.b = b; this.N = corpus.length;
    this.inverted = new Map();          // term -> [[docIdx, tf], ...]
    this.docLen = new Float32Array(this.N);
    this.idf = new Map();
    const df = new Map();
    corpus.forEach((doc, idx) => {
      const toks = tokenize(doc);
      this.docLen[idx] = toks.length;
      const tf = new Map();
      for (const t of toks) tf.set(t, (tf.get(t) || 0) + 1);
      for (const [t, f] of tf) {
        if (!this.inverted.has(t)) this.inverted.set(t, []);
        this.inverted.get(t).push([idx, f]);
        df.set(t, (df.get(t) || 0) + 1);
      }
    });
    let sum = 0; for (const l of this.docLen) sum += l;
    this.avgdl = this.N ? sum / this.N : 0;
    for (const [t, d] of df) this.idf.set(t, Math.log(1 + (this.N - d + 0.5) / (d + 0.5)));
  }
  search(query, k) {
    const scores = new Float32Array(this.N);
    for (const t of tokenize(query)) {
      const postings = this.inverted.get(t);
      if (!postings) continue;
      const idf = this.idf.get(t);
      for (const [doc, f] of postings) {
        const denom = f + this.k1 * (1 - this.b + this.b * this.docLen[doc] / this.avgdl);
        scores[doc] += idf * (f * (this.k1 + 1)) / denom;
      }
    }
    return topK(scores, k).filter(([, s]) => s > 0);
  }
}

// ---------------------------------------------------------------------------
// Dense search + fusion + MMR - ported from src/{vector_store,retriever}.py
// ---------------------------------------------------------------------------
function topK(scores, k) {
  const idx = Array.from(scores.keys());
  idx.sort((a, b) => scores[b] - scores[a]);
  return idx.slice(0, k).map(i => [i, scores[i]]);
}
function dot(a, aOff, b, bOff) {
  let s = 0; for (let d = 0; d < DIM; d++) s += a[aOff + d] * b[bOff + d]; return s;
}
function denseSearch(qvec, k) {
  const scores = new Float32Array(N);
  for (let r = 0; r < N; r++) scores[r] = dot(qvec, 0, MAT, r * DIM);
  return topK(scores, k);
}
function rrf(lists, k = 60) {
  const fused = new Map();
  for (const lst of lists)
    lst.forEach(([id], rank) => fused.set(id, (fused.get(id) || 0) + 1 / (k + rank)));
  return fused;
}
function mmr(qvec, candIds, n, lambda = 0.7) {
  const selected = [], remaining = [...candIds];
  const rel = new Map(candIds.map(i => [i, dot(qvec, 0, MAT, i * DIM)]));
  while (remaining.length && selected.length < n) {
    let best = null, bestScore = -1e9;
    for (const i of remaining) {
      let redundancy = 0;
      for (const j of selected) redundancy = Math.max(redundancy, dot(MAT, i * DIM, MAT, j * DIM));
      const score = lambda * rel.get(i) - (1 - lambda) * redundancy;
      if (score > bestScore) { bestScore = score; best = i; }
    }
    selected.push(best);
    remaining.splice(remaining.indexOf(best), 1);
  }
  return selected;
}

async function embedQuery(text) {
  // CLS pooling - must match build_embeddings.mjs (BGE is a CLS-pooled model).
  const out = await extractor(QUERY_PREFIX + text, { pooling: 'cls', normalize: true });
  return out.data; // Float32Array(384), normalised
}

// ---------------------------------------------------------------------------
// Web Worker: writer (text-generation) + Whisper (speech-to-text), off-thread.
// ---------------------------------------------------------------------------
let worker = null, loadHandlers = null, genHandlers = null;
let genReady = false, asrReady = false, asrLoadHandlers = null, asrHandlers = null;
const isProg = d => d?.status === 'progress' && /onnx|model|data/i.test(d.file || '');

function ensureWorker() {
  if (worker) return;
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (e) => {
    const m = e.data;
    switch (m.type) {
      case 'progress': if (isProg(m.data)) loadHandlers?.onProgress?.(Math.round(m.data.progress || 0)); break;
      case 'ready': genReady = true; loadHandlers?.resolve?.(); loadHandlers = null; break;
      case 'token': genHandlers?.onToken?.(m.token); break;
      case 'done': genHandlers?.resolve?.(m.text); genHandlers = null; break;
      case 'asr-progress': if (isProg(m.data)) asrLoadHandlers?.onProgress?.(Math.round(m.data.progress || 0)); break;
      case 'asr-ready': asrReady = true; asrLoadHandlers?.resolve?.(); asrLoadHandlers = null; break;
      case 'transcript': asrHandlers?.resolve?.(m.text); asrHandlers = null; break;
      case 'error':
        if (m.op === 'load' || m.op === 'generate') {
          (genHandlers?.reject || loadHandlers?.reject)?.(new Error(m.error)); genHandlers = null; loadHandlers = null;
        } else {
          (asrHandlers?.reject || asrLoadHandlers?.reject)?.(new Error(m.error)); asrHandlers = null; asrLoadHandlers = null;
        }
        break;
    }
  };
}
function loadWriter(onProgress) {
  return new Promise((resolve, reject) => {
    ensureWorker();
    if (genReady) return resolve();
    loadHandlers = { onProgress, resolve, reject };
    worker.postMessage({ type: 'load', model: GEN_MODEL });
  });
}
function loadASR(onProgress) {
  return new Promise((resolve, reject) => {
    ensureWorker();
    if (asrReady) return resolve();
    asrLoadHandlers = { onProgress, resolve, reject };
    worker.postMessage({ type: 'load-asr', model: ASR_MODEL });
  });
}
function transcribe(audio) {
  return new Promise((resolve, reject) => {
    asrHandlers = { resolve, reject };
    worker.postMessage({ type: 'transcribe', audio }, [audio.buffer]);
  });
}
function streamAnswer(question, retrieved, onToken) {
  return new Promise((resolve, reject) => {
    const context = retrieved.map((r, i) =>
      `[${i + 1}] (Source: ${r.doc_title}, ${pages(r)})\n${r.text}`).join('\n\n');
    genHandlers = { onToken, resolve, reject };
    worker.postMessage({ type: 'generate', messages: [
      { role: 'system', content: GEN_SYSTEM },
      { role: 'user', content: `Policy excerpts:\n\n${context}\n\n---\nQuestion: ${question}` },
    ] });
  });
}

// Retrieval only (shared by the streaming chat UI and the non-streaming ask()).
async function retrieve(question, topN = 6, fetchK = 25) {
  const qvec = await embedQuery(question);
  const dense = denseSearch(qvec, fetchK);
  const lexical = BM25.search(question, fetchK);
  const confidence = dense.length ? dense[0][1] : 0;
  if (!dense.length || confidence < ABSTAIN_THRESHOLD)
    return { abstained: true, confidence, retrieved: [], citations: [] };
  const denseScore = new Map(dense), bmScore = new Map(lexical);
  const fused = rrf([dense, lexical]);
  const candidates = [...fused.keys()].sort((a, b) => fused.get(b) - fused.get(a))
    .slice(0, Math.max(topN * 3, 15));
  const chosen = mmr(qvec, candidates, topN);
  const retrieved = chosen.map((i, k) => {
    const c = CHUNKS[i];
    return {
      n: k + 1, doc_title: c.doc_title, page_start: c.page_start, page_end: c.page_end,
      dense: +(denseScore.get(i) || 0).toFixed(3), bm25: +(bmScore.get(i) || 0).toFixed(3),
      rrf: +(fused.get(i) || 0).toFixed(4),
      text: c.text, preview: c.text.length > 240 ? c.text.slice(0, 240) + '…' : c.text,
    };
  });
  const citations = retrieved.map(r => ({ n: r.n, doc_title: r.doc_title, page_start: r.page_start, page_end: r.page_end }));
  return { abstained: false, confidence, retrieved, citations };
}

// Non-streaming answer (hosted endpoint / extractive / in-browser collected to text).
// Used by the console API window.policyDesk.ask(); the chat UI streams (see handle()).
async function ask(question) {
  const r = await retrieve(question);
  if (r.abstained) return { answer: REFUSAL, mode: 'abstained', confidence: r.confidence, citations: [], retrieved: [] };
  const { retrieved, citations, confidence } = r;

  if (LLM_ENABLED) {
    try {
      const resp = await fetch('api/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, contexts: retrieved.map(x => ({
          doc_title: x.doc_title, page_start: x.page_start, page_end: x.page_end, text: x.text })) }),
      });
      if (resp.ok) {
        const j = await resp.json();
        if (j.mode === 'abstained') return { answer: REFUSAL, mode: 'abstained', confidence, citations: [], retrieved };
        return { answer: j.answer, mode: 'written', confidence, citations, retrieved };
      }
    } catch (e) { /* fall through */ }
  }
  if (WRITTEN_MODE && genReady) {
    try {
      const text = await streamAnswer(question, retrieved, () => {});
      if (text) return { answer: text.trim(), mode: 'written', confidence, citations, retrieved };
    } catch (e) { /* fall through */ }
  }
  const lines = ['Based on the policy documents:\n'];
  retrieved.slice(0, 5).forEach((x, k) => {
    let s = x.text.trim();
    if (s.length > 480) s = s.slice(0, 480).replace(/\s+\S*$/, '') + '…';
    lines.push(`- ${s} [${k + 1}]`);
  });
  return { answer: lines.join('\n'), mode: 'extractive', confidence, citations, retrieved };
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------
const chat = document.getElementById('chat');
const q = document.getElementById('q');
const send = document.getElementById('send');
const statusEl = document.getElementById('status');

const esc = s => s.replace(/[&<>]/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[m]));
function el(t, c, html) { const e = document.createElement(t); if (c) e.className = c; if (html != null) e.innerHTML = html; return e; }
const scroll = () => { chat.scrollTop = chat.scrollHeight; };

function addUser(text) {
  const m = el('div', 'msg user'); m.appendChild(el('div', 'who', 'You'));
  m.appendChild(el('div', 'bubble', esc(text))); chat.appendChild(m); scroll();
}
function addThinking() {
  const m = el('div', 'msg bot'); m.id = 'thinking'; m.appendChild(el('div', 'who', 'Assistant'));
  m.appendChild(el('div', 'bubble', '<span class="dots"><span>●</span><span>●</span><span>●</span></span>'));
  chat.appendChild(m); scroll();
}
function pages(c) { return c.page_start === c.page_end ? 'p.' + c.page_start : 'pp.' + c.page_start + '-' + c.page_end; }
// Append citations + mode/confidence + the collapsible sources to a bot message.
function renderMeta(m, res) {
  if (res.citations.length) {
    const c = el('div', 'cites');
    res.citations.forEach(ci => c.appendChild(el('span', 'cite', `[${ci.n}] ${esc(ci.doc_title)} · ${pages(ci)}`)));
    m.appendChild(c);
  }
  m.appendChild(el('div', 'tags', `<span class="tag-${res.mode}">mode: ${res.mode}</span> · <span>confidence: ${res.confidence.toFixed(3)}</span>`));
  if (res.retrieved.length) {
    const d = el('details'); d.appendChild(el('summary', null, `▸ sources used (${res.retrieved.length})`));
    res.retrieved.forEach(r => {
      const rd = el('div', 'ret');
      rd.appendChild(el('div', null, `<span class="src">[${r.n}] ${esc(r.doc_title)} · ${pages(r)}</span><span class="sc">dense ${r.dense} · bm25 ${r.bm25} · rrf ${r.rrf}</span>`));
      rd.appendChild(el('div', 'pv', esc(r.preview))); d.appendChild(rd);
    });
    m.appendChild(d);
  }
  scroll();
}
function addBot(res) {
  document.getElementById('thinking')?.remove();
  const m = el('div', 'msg bot'); m.appendChild(el('div', 'who', 'Assistant'));
  m.appendChild(el('div', 'bubble' + (res.mode === 'abstained' ? ' abstained' : ''), esc(res.answer)));
  renderMeta(m, res);
  chat.appendChild(m); scroll();
}
// Streamed reply for in-browser written mode: tokens land in the bubble live.
async function streamReply(question, r) {
  document.getElementById('thinking')?.remove();
  const m = el('div', 'msg bot'); m.appendChild(el('div', 'who', 'Assistant'));
  const bubble = el('div', 'bubble', '<span class="dots"><span>●</span><span>●</span><span>●</span></span>');
  m.appendChild(bubble); chat.appendChild(m); scroll();
  let acc = '';
  const onToken = t => { acc += t; bubble.textContent = acc; scroll(); };
  let text;
  try { text = await streamAnswer(question, r.retrieved, onToken); }
  catch (e) { text = acc; }
  bubble.textContent = (text || acc).trim() || REFUSAL;
  renderMeta(m, { mode: 'written', confidence: r.confidence, citations: r.citations, retrieved: r.retrieved });
}
function addSystem(text) {
  const m = el('div', 'msg sys'); m.appendChild(el('div', 'sysmsg', esc(text)));
  chat.appendChild(m); scroll();
}
function refreshDocs() {
  const total = new Set(CHUNKS.map(c => c.doc_id)).size;
  const up = new Set(CHUNKS.filter(c => c.doc_id.startsWith('upload:')).map(c => c.doc_id)).size;
  document.getElementById('b-docs').textContent =
    up ? `${total} policy documents (${up} uploaded)` : `${total} policy documents`;
}

async function handle(text) {
  if (!text || send.disabled) return;
  q.value = ''; send.disabled = true; q.disabled = true;
  addUser(text); addThinking();
  try {
    // In-browser written mode streams tokens into the bubble; everything else
    // (hosted endpoint, extractive) returns a full answer at once.
    if (WRITTEN_MODE && genReady && !LLM_ENABLED) {
      const r = await retrieve(text);
      if (r.abstained) addBot({ answer: REFUSAL, mode: 'abstained', confidence: r.confidence, citations: [], retrieved: [] });
      else await streamReply(text, r);
    } else {
      addBot(await ask(text));
    }
  } catch (e) {
    document.getElementById('thinking')?.remove();
    addBot({ answer: 'Error: ' + e.message, mode: 'abstained', confidence: 0, citations: [], retrieved: [] });
  }
  send.disabled = false; q.disabled = false; q.focus();
}

// ---------------------------------------------------------------------------
// Upload: add user policy PDFs to the in-browser index.
// extract (pdf.js, OCR fallback) -> chunk -> embed -> merge. All in the browser.
// ---------------------------------------------------------------------------
function cleanText(t) { return t.replace(/-\n(?=[a-z])/g, '').replace(/\s+/g, ' ').trim(); }
function splitSentences(t) {
  return t.split(/(?<=[.!?])\s+(?=["'(\[A-Z0-9])/).map(s => s.trim()).filter(Boolean);
}
function wc(s) { return s.split(/\s+/).filter(Boolean).length; }
function chunkPages(docId, docTitle, pages, startId, target = 110, overlap = 25) {
  const units = [];
  for (const p of pages) for (const s of splitSentences(p.text)) units.push({ s, page: p.page });
  const out = []; let cur = [], words = 0, cid = startId;
  const flush = () => {
    if (!cur.length) return;
    out.push({ id: cid++, doc_id: docId, doc_title: docTitle,
      page_start: Math.min(...cur.map(u => u.page)), page_end: Math.max(...cur.map(u => u.page)),
      text: cur.map(u => u.s).join(' ') });
  };
  for (const u of units) {
    cur.push(u); words += wc(u.s);
    if (words >= target) {
      flush();
      let tail = [], tw = 0;
      for (let i = cur.length - 1; i >= 0; i--) { tw += wc(cur[i].s); tail.unshift(cur[i]); if (tw >= overlap) break; }
      cur = tail; words = tail.reduce((a, u) => a + wc(u.s), 0);
    }
  }
  flush();
  return out;
}

let _ocrWorker = null;
async function ocrPage(page) {                 // OCR a scanned page (lazy-loads tesseract.js)
  try {
    const T = await import('https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.esm.min.js');
    if (!_ocrWorker) _ocrWorker = await T.createWorker('eng');
    const vp = page.getViewport({ scale: 2 });
    const cv = document.createElement('canvas'); cv.width = vp.width; cv.height = vp.height;
    await page.render({ canvasContext: cv.getContext('2d'), viewport: vp }).promise;
    const { data } = await _ocrWorker.recognize(cv);
    return data.text || '';
  } catch (e) { return ''; }
}
async function extractPdf(file) {
  const pdf = await pdfjsLib.getDocument({ data: await file.arrayBuffer() }).promise;
  const pages = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    let txt = (await page.getTextContent()).items.map(it => it.str).join(' ');
    if (cleanText(txt).length < 40) {          // no text layer -> scanned -> OCR
      statusEl.textContent = `running OCR on ${file.name} (page ${i})…`;
      const ocr = await ocrPage(page);
      if (ocr.length > txt.length) txt = ocr;
    }
    txt = cleanText(txt);
    if (txt.length >= 40) pages.push({ page: i, text: txt });
  }
  return pages;
}
async function embedAndAppend(newChunks) {
  const B = 8, vecs = [];
  for (let i = 0; i < newChunks.length; i += B) {
    const batch = newChunks.slice(i, i + B).map(c => c.text);
    const out = await extractor(batch, { pooling: 'cls', normalize: true });
    for (let j = 0; j < batch.length; j++) vecs.push(out.data.slice(j * DIM, (j + 1) * DIM));
    statusEl.textContent = `embedding ${Math.min(i + B, newChunks.length)}/${newChunks.length}…`;
  }
  const oldN = N, merged = new Float32Array((N + newChunks.length) * DIM);
  merged.set(MAT, 0);
  newChunks.forEach((c, i) => { c.id = oldN + i; merged.set(vecs[i], (oldN + i) * DIM); });
  MAT = merged; CHUNKS = CHUNKS.concat(newChunks); N = CHUNKS.length;
  BM25 = new BM25Index(CHUNKS.map(c => c.text));   // rebuild lexical index over all chunks
}

let uploading = false;
async function processFiles(files) {
  if (uploading || !extractor) return;
  const pdfs = [...files].filter(f => /\.pdf$/i.test(f.name) || f.type === 'application/pdf');
  if (!pdfs.length) return;
  uploading = true; uploadBtn.disabled = true; send.disabled = true; q.disabled = true;
  try {
    for (const f of pdfs) {
      statusEl.textContent = `reading ${f.name}…`;
      const pages = await extractPdf(f);
      if (!pages.length) { addSystem(`Could not read text from "${f.name}". It may be an unsupported or empty scan.`); continue; }
      const title = f.name.replace(/\.pdf$/i, '');
      const chunks = chunkPages('upload:' + title, title, pages, N);
      await embedAndAppend(chunks);
      addSystem(`Added "${f.name}": ${chunks.length} passage${chunks.length > 1 ? 's' : ''} from ${pages.length} page${pages.length > 1 ? 's' : ''}. You can ask about it now.`);
    }
    refreshDocs();
  } catch (e) {
    addSystem('Upload failed: ' + e.message);
  } finally {
    statusEl.textContent = ''; uploading = false;
    uploadBtn.disabled = false; send.disabled = false; q.disabled = false;
  }
}

async function probeLLM() {
  // If a hosted GLM endpoint is configured, written answers come from there.
  try {
    const r = await fetch('api/chat');
    if (r.ok) LLM_ENABLED = !!(await r.json()).enabled;
  } catch (e) { /* static-only deploy: client-side only */ }
}

async function probeVoice() {
  // If a hosted Groq Whisper endpoint is configured, the mic uses it.
  try {
    const r = await fetch('api/voice');
    if (r.ok) VOICE_ENABLED = !!(await r.json()).enabled;
  } catch (e) { /* no voice on this deploy */ }
}

async function boot() {
  await loadIndex();
  await Promise.all([probeLLM(), probeVoice()]);
  statusEl.textContent = 'loading embedding model (one-time, ~30MB)…';
  extractor = await pipeline('feature-extraction', META.model, {
    dtype: META.dtype,
    progress_callback: (p) => {
      if (p.status === 'progress' && p.file && p.file.includes('onnx'))
        statusEl.textContent = `downloading model… ${Math.round(p.progress || 0)}%`;
    },
  });
  statusEl.textContent = '';
  q.disabled = false; send.disabled = false; uploadBtn.disabled = false;
  if (micBtn) micBtn.disabled = !VOICE_ENABLED;  // mic only when Groq voice is on
  q.placeholder = 'Ask about a policy - coverage, limits, claims, fees…'; q.focus();
  // Writer toggle: if a hosted endpoint is on, answers are already written; else
  // let the user turn on in-browser generation.
  if (LLM_ENABLED) { writerToggle.textContent = 'Written answers: On'; writerToggle.disabled = true; }
  else writerToggle.disabled = false;
}

// Small public API (handy for scripting / testing from the console).
window.policyDesk = { ask, addFiles: processFiles, loadASR, transcribe };

send.onclick = () => handle(q.value.trim());
q.onkeydown = e => { if (e.key === 'Enter') handle(q.value.trim()); };
document.querySelectorAll('.chip').forEach(c => c.onclick = () => handle(c.textContent));

const writerToggle = document.getElementById('writer-toggle');
writerToggle.onclick = async () => {
  if (LLM_ENABLED) return;                       // hosted endpoint already writes
  if (!WRITTEN_MODE) {
    writerToggle.disabled = true; writerToggle.textContent = 'Loading writer…';
    try {
      await loadWriter(pct => { statusEl.textContent = `loading writer model… ${pct}%`; });
      WRITTEN_MODE = true; writerToggle.textContent = 'Written answers: On';
      statusEl.textContent = '';
      addSystem('Written answers are on. Replies are composed by a small open model running entirely in your browser, grounded in the policy documents.');
    } catch (e) {
      statusEl.textContent = ''; writerToggle.textContent = 'Written answers: Off';
      addSystem('Could not load the writer model: ' + e.message);
    } finally { writerToggle.disabled = false; }
  } else {
    WRITTEN_MODE = false; writerToggle.textContent = 'Written answers: Off';
  }
};

const uploadBtn = document.getElementById('upload-btn');
const fileInput = document.getElementById('file');
uploadBtn.onclick = () => fileInput.click();
fileInput.onchange = () => { processFiles(fileInput.files); fileInput.value = ''; };
['dragenter', 'dragover'].forEach(ev => document.addEventListener(ev, e => {
  if (e.dataTransfer?.types?.includes('Files')) { e.preventDefault(); document.body.classList.add('dragging'); }
}));
document.addEventListener('dragleave', e => { if (e.relatedTarget === null) document.body.classList.remove('dragging'); });
document.addEventListener('drop', e => {
  e.preventDefault(); document.body.classList.remove('dragging');
  if (e.dataTransfer?.files?.length) processFiles(e.dataTransfer.files);
});

// ---------------------------------------------------------------------------
// Voice input: record from the mic, transcribe server-side with Groq Whisper
// (whisper-large-v3), then run the question through the normal grounded pipeline.
// ---------------------------------------------------------------------------
const micBtn = document.getElementById('mic');
let recording = false, mediaRecorder = null, audioChunks = [];

function setMic(state) {
  if (micBtn) micBtn.dataset.state = state;
}
function blobToBase64(blob) {                          // -> base64 string (no data: prefix)
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onerror = reject;
    fr.onload = () => resolve(String(fr.result).split(',')[1] || '');
    fr.readAsDataURL(blob);
  });
}
async function startRecording() {
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
  catch (e) { addSystem('Microphone access was blocked. Allow it to ask by voice.'); return; }
  audioChunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = e => { if (e.data.size) audioChunks.push(e.data); };
  mediaRecorder.onstop = async () => {
    stream.getTracks().forEach(t => t.stop());
    recording = false; setMic('transcribing'); statusEl.textContent = 'transcribing…';
    try {
      const mime = mediaRecorder.mimeType || 'audio/webm';
      const audio = await blobToBase64(new Blob(audioChunks, { type: mime }));
      const resp = await fetch('api/voice', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audio, mime }),
      });
      const j = await resp.json();
      statusEl.textContent = ''; setMic('idle');
      if (!resp.ok) { addSystem('Could not transcribe: ' + (j.error || resp.status)); return; }
      if (j.text && j.text.trim()) { q.value = j.text.trim(); handle(j.text.trim()); }
      else addSystem('I did not catch that. Please try again.');
    } catch (e) { statusEl.textContent = ''; setMic('idle'); addSystem('Could not transcribe: ' + e.message); }
  };
  mediaRecorder.start();
  recording = true; setMic('recording');
}
if (micBtn) micBtn.onclick = () => { if (recording) mediaRecorder?.stop(); else startRecording(); };

boot().catch(e => { statusEl.textContent = 'failed to load: ' + e.message; });
