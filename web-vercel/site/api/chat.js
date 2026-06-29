// Vercel serverless function: Claude (Anthropic) grounded generation layer.
//
// Calls the Anthropic Messages API:
//   ANTHROPIC_API_KEY = sk-ant-...        (required to enable written answers)
//   ANTHROPIC_MODEL   = claude-sonnet-4-6 (optional override)
//
// The browser does retrieval and sends the top policy passages here. Claude answers
// using ONLY those passages, cites them, and REFUSES when they don't actually
// answer - the grounding judgment a bi-encoder can't make.
//
// If ANTHROPIC_API_KEY is not configured, GET reports disabled and the frontend
// stays in extractive / in-browser mode, so the site works with zero secrets.

const ANTHROPIC_URL = 'https://api.anthropic.com/v1/messages';
const MODEL = process.env.ANTHROPIC_MODEL || 'claude-sonnet-4-6';
const REFUSAL = "I couldn't find that in the policy documents.";

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
  const key = process.env.ANTHROPIC_API_KEY;

  if (req.method === 'GET') {
    return res.status(200).json({ enabled: !!key, model: key ? MODEL : null });
  }
  if (req.method !== 'POST') return res.status(405).json({ error: 'method not allowed' });
  if (!key) return res.status(501).json({ error: 'ANTHROPIC_API_KEY not configured' });

  try {
    const { question, contexts } = req.body || {};
    if (!question || !Array.isArray(contexts) || !contexts.length)
      return res.status(400).json({ error: 'question and contexts required' });

    const blocks = contexts.map((c, i) => {
      const pages = c.page_start === c.page_end ? `p.${c.page_start}` : `pp.${c.page_start}-${c.page_end}`;
      return `[${i + 1}] (Source: ${c.doc_title}, ${pages})\n${c.text}`;
    }).join('\n\n');

    const r = await fetch(ANTHROPIC_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': key,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: 1024,
        temperature: 0,
        system: SYSTEM,
        messages: [
          { role: 'user', content: `Policy excerpts:\n\n${blocks}\n\n---\nQuestion: ${question}` },
        ],
      }),
    });
    if (!r.ok) {
      const detail = await r.text();
      return res.status(502).json({ error: `Claude ${r.status}`, detail: detail.slice(0, 300) });
    }
    const data = await r.json();
    const text = (data?.content?.[0]?.text || '').trim();
    if (!text || text === REFUSAL) return res.status(200).json({ answer: REFUSAL, mode: 'abstained' });
    return res.status(200).json({ answer: text, mode: 'claude' });
  } catch (e) {
    return res.status(500).json({ error: String(e?.message || e) });
  }
}
