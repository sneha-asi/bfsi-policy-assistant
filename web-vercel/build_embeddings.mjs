/**
 * Build-time passage embedding for the static (client-side) RAG site.
 *
 * The deployed site embeds the *query* in the browser with Transformers.js. To
 * guarantee query and passage vectors come from an identical implementation, we
 * embed the passages HERE with the SAME library + model + dtype, and record the
 * exact model/dtype in meta.json so the browser reads it back (no drift possible).
 *
 * Outputs (static assets, served by Vercel):
 *   site/data/chunks.json     trimmed chunk records (text + source + pages)
 *   site/data/embeddings.bin  Float32 (N x 384) row-major, L2-normalised
 *   site/data/meta.json       { n_chunks, dim, model, dtype }
 */
import { pipeline } from '@huggingface/transformers';
import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dir = dirname(fileURLToPath(import.meta.url));
const SRC_CHUNKS = join(__dir, '..', 'data', 'index', 'chunks.json');
const OUT_DIR = join(__dir, 'site', 'data');

const MODEL = 'Xenova/bge-small-en-v1.5';
const QUERY_PREFIX = 'Represent this sentence for searching relevant passages: ';
const DIM = 384;
const BATCH = 32;

async function makeExtractor() {
  // q8 (~33MB browser download) is fine once pooling is correct; fp32 fallback.
  for (const dtype of ['q8', 'fp32']) {
    try {
      const ex = await pipeline('feature-extraction', MODEL, { dtype });
      console.log(`Loaded ${MODEL} (dtype=${dtype})`);
      return { ex, dtype };
    } catch (e) {
      console.log(`dtype=${dtype} failed (${e.message}); trying next…`);
    }
  }
  throw new Error('could not load embedding model');
}

async function embed(ex, texts) {
  // BGE uses CLS pooling (the [CLS] token), NOT mean - using mean degrades the
  // vectors badly and collapses in/out separation. This must match app.js.
  const out = await ex(texts, { pooling: 'cls', normalize: true });
  return out.data; // flat Float32Array length texts.length*DIM
}

async function main() {
  const chunks = JSON.parse(readFileSync(SRC_CHUNKS, 'utf8'));
  const N = chunks.length;
  console.log(`Embedding ${N} chunks…`);
  const { ex, dtype } = await makeExtractor();

  const mat = new Float32Array(N * DIM);
  for (let i = 0; i < N; i += BATCH) {
    const batch = chunks.slice(i, i + BATCH).map(c => c.text);
    const data = await embed(ex, batch);
    mat.set(data, i * DIM);
    process.stdout.write(`\r  embedded ${Math.min(i + BATCH, N)}/${N}`);
  }
  process.stdout.write('\n');

  mkdirSync(OUT_DIR, { recursive: true });
  const trimmed = chunks.map(c => ({
    id: c.id, doc_id: c.doc_id, doc_title: c.doc_title,
    page_start: c.page_start, page_end: c.page_end, text: c.text,
  }));
  writeFileSync(join(OUT_DIR, 'chunks.json'), JSON.stringify(trimmed));
  writeFileSync(join(OUT_DIR, 'embeddings.bin'), Buffer.from(mat.buffer));
  writeFileSync(join(OUT_DIR, 'meta.json'),
    JSON.stringify({ n_chunks: N, dim: DIM, model: MODEL, dtype }, null, 2));
  console.log(`Wrote ${N} chunks, ${(mat.byteLength / 1e6).toFixed(1)}MB embeddings.`);

  // --- Re-calibrate abstention threshold for THESE embeddings ---
  const IN = [
    'What is the waiting period for pre-existing diseases?',
    'How long does claim settlement take?',
    'What is the home loan eligibility?',
    'What are the credit card late payment fees?',
    'What is the minimum balance for a savings account?',
    'What is the suicide clause in life insurance?',
    // the UI's suggested chips MUST clear the threshold:
    'What is the daily ATM withdrawal limit?',
    'How is No Claim Bonus calculated on motor insurance?',
  ];
  const OUT = [
    'Who won the 2018 FIFA World Cup?', 'What is the boiling point of water?',
    'What is the capital of Australia?', 'How do I bake sourdough bread?',
    'Who painted the Mona Lisa?', 'How tall is Mount Everest?',
  ];
  const maxCos = async (q) => {
    const v = await embed(ex, [QUERY_PREFIX + q]); // normalized 384
    let best = -1;
    for (let r = 0; r < N; r++) {
      let s = 0; const off = r * DIM;
      for (let d = 0; d < DIM; d++) s += v[d] * mat[off + d];
      if (s > best) best = s;
    }
    return best;
  };
  const inS = []; for (const q of IN) inS.push(await maxCos(q));
  const outS = []; for (const q of OUT) outS.push(await maxCos(q));
  const inMin = Math.min(...inS), outMax = Math.max(...outS);
  console.log('\nCalibration (max cosine):');
  console.log('  IN  min=%s mean=%s', inMin.toFixed(3), (inS.reduce((a,b)=>a+b)/inS.length).toFixed(3));
  console.log('  OUT max=%s mean=%s', outMax.toFixed(3), (outS.reduce((a,b)=>a+b)/outS.length).toFixed(3));
  console.log('  Suggested threshold (gap midpoint): %s', ((inMin + outMax) / 2).toFixed(3));
}

main().catch(e => { console.error(e); process.exit(1); });
