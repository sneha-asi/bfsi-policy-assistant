// Vercel serverless function: voice transcription via Groq Whisper.
//
// The browser records mic audio, base64-encodes it, and POSTs it here. We forward
// it to Groq's OpenAI-compatible audio endpoint (whisper-large-v3 - accurate and
// fast) and return the transcript. The frontend then runs that text through the
// normal grounded retrieval + GLM pipeline.
//
//   GROQ_API_KEY   = <provider key>
//   GROQ_ASR_MODEL = whisper-large-v3   (optional override)
//
// GET reports whether voice is enabled so the UI can show/hide the mic button.

const GROQ_URL = 'https://api.groq.com/openai/v1/audio/transcriptions';
const MODEL = process.env.GROQ_ASR_MODEL || 'whisper-large-v3';

export const config = { api: { bodyParser: { sizeLimit: '8mb' } } };

export default async function handler(req, res) {
  const key = process.env.GROQ_API_KEY;

  if (req.method === 'GET') {
    return res.status(200).json({ enabled: !!key, model: key ? MODEL : null });
  }
  if (req.method !== 'POST') return res.status(405).json({ error: 'method not allowed' });
  if (!key) return res.status(501).json({ error: 'GROQ_API_KEY not configured' });

  try {
    const { audio, mime } = req.body || {};
    if (!audio) return res.status(400).json({ error: 'audio (base64) required' });

    const buffer = Buffer.from(audio, 'base64');
    const form = new FormData();
    form.append('file', new Blob([buffer], { type: mime || 'audio/webm' }), 'audio.webm');
    form.append('model', MODEL);
    form.append('language', 'en');
    form.append('response_format', 'json');

    const r = await fetch(GROQ_URL, {
      method: 'POST',
      headers: { Authorization: `Bearer ${key}` },
      body: form,
    });
    if (!r.ok) {
      const detail = await r.text();
      return res.status(502).json({ error: `Groq ${r.status}`, detail: detail.slice(0, 300) });
    }
    const data = await r.json();
    return res.status(200).json({ text: (data?.text || '').trim() });
  } catch (e) {
    return res.status(500).json({ error: String(e?.message || e) });
  }
}
