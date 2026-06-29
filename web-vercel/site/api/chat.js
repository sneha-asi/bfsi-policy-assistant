// Vercel serverless function: optional GLM (open-model) generation layer.
//
// Calls any OpenAI-compatible chat endpoint, so it works with a hosted GLM:
//   LLM_BASE_URL = https://openrouter.ai/api/v1
//   LLM_MODEL    = z-ai/glm-4.6        (any GLM variant the provider exposes)
//   LLM_API_KEY  = <provider key>
//
// The browser does retrieval and sends the top policy passages here. GLM answers
// using ONLY those passages, cites them, and REFUSES when they don't actually
// answer - the grounding judgment a bi-encoder can't make.
//
// If LLM_API_KEY is not configured, GET reports disabled and the frontend stays in
// extractive mode, so the site works with zero secrets.

const REFUSAL = "I couldn't find that in the policy documents.";
const BASE_URL = (process.env.LLM_BASE_URL || 'https://openrouter.ai/api/v1').replace(/\/$/, '');
const MODEL = process.env.LLM_MODEL || 'z-ai/glm-4.6';

const SYSTEM = `You are a warm, helpful policy advisor at a bank/insurance company, like a \
knowledgeable support agent speaking to a customer. Answer in plain, friendly, natural \
language, the way a real person would explain it.

How to answer:
- Lead with the direct answer in one sentence, then briefly explain it in everyday terms.
- Keep it conversational and concise (about 2 to 5 sentences). Avoid bullet-point dumps and \
jargon unless the customer asks for a list.
- Always include the exact figures that matter (amounts, percentages, waiting periods, timelines).
- Quietly cite the excerpt number(s) you used at the end of the relevant sentence, like [1].

Hard rules:
- Use ONLY the information in the policy excerpts below. Never use outside knowledge or guess.
- If the excerpts do not contain the answer, say so politely in your own words and suggest the \
customer confirm with support. Never invent policy terms or numbers.`;

export default async function handler(req, res) {
  const key = process.env.LLM_API_KEY;

  if (req.method === 'GET') {
    return res.status(200).json({ enabled: !!key, model: key ? MODEL : null });
  }
  if (req.method !== 'POST') return res.status(405).json({ error: 'method not allowed' });
  if (!key) return res.status(501).json({ error: 'LLM_API_KEY not configured' });

  try {
    const { question, contexts } = req.body || {};
    if (!question || !Array.isArray(contexts) || !contexts.length)
      return res.status(400).json({ error: 'question and contexts required' });

    const blocks = contexts.map((c, i) => {
      const pages = c.page_start === c.page_end ? `p.${c.page_start}` : `pp.${c.page_start}-${c.page_end}`;
      return `[${i + 1}] (Source: ${c.doc_title}, ${pages})\n${c.text}`;
    }).join('\n\n');

    const r = await fetch(`${BASE_URL}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
      body: JSON.stringify({
        model: MODEL,
        temperature: 0,
        max_tokens: 1024,
        messages: [
          { role: 'system', content: SYSTEM },
          { role: 'user', content: `Policy excerpts:\n\n${blocks}\n\n---\nQuestion: ${question}` },
        ],
      }),
    });
    if (!r.ok) {
      const detail = await r.text();
      return res.status(502).json({ error: `LLM provider ${r.status}`, detail: detail.slice(0, 300) });
    }
    const data = await r.json();
    const text = (data?.choices?.[0]?.message?.content || '').trim();
    if (!text || text === REFUSAL) return res.status(200).json({ answer: REFUSAL, mode: 'abstained' });
    return res.status(200).json({ answer: text, mode: 'glm' });
  } catch (e) {
    return res.status(500).json({ error: String(e?.message || e) });
  }
}
