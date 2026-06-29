// Web Worker: runs the open models off the main thread.
//  - text-generation (writer) for written answers, with token streaming.
//  - automatic-speech-recognition (Whisper) for voice questions.
// Both are lazy-loaded on first use. Nothing leaves the browser.
import { pipeline, TextStreamer, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6';

env.allowLocalModels = false;
env.useBrowserCache = true;

let gen = null;   // text-generation pipeline
let asr = null;   // speech-recognition pipeline

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'load') {
      for (const opts of [{ dtype: 'q4f16', device: 'webgpu' }, { dtype: 'q4' }]) {
        try {
          gen = await pipeline('text-generation', msg.model, {
            ...opts, progress_callback: (p) => self.postMessage({ type: 'progress', data: p }),
          });
          break;
        } catch (_) { /* next */ }
      }
      if (!gen) throw new Error('writer model failed to load');
      self.postMessage({ type: 'ready' });

    } else if (msg.type === 'generate') {
      const streamer = new TextStreamer(gen.tokenizer, {
        skip_prompt: true, skip_special_tokens: true,
        callback_function: (t) => self.postMessage({ type: 'token', token: t }),
      });
      const out = await gen(msg.messages, {
        max_new_tokens: 220, return_full_text: false,
        do_sample: true, temperature: 0.3, top_p: 0.9,
        repetition_penalty: 1.3, no_repeat_ngram_size: 3, streamer,
      });
      const gt = out[0].generated_text;
      const text = Array.isArray(gt) ? gt[gt.length - 1].content : String(gt);
      self.postMessage({ type: 'done', text });

    } else if (msg.type === 'load-asr') {
      for (const opts of [{ dtype: 'q8' }, {}]) {
        try {
          asr = await pipeline('automatic-speech-recognition', msg.model, {
            ...opts, progress_callback: (p) => self.postMessage({ type: 'asr-progress', data: p }),
          });
          break;
        } catch (_) { /* next */ }
      }
      if (!asr) throw new Error('speech model failed to load');
      self.postMessage({ type: 'asr-ready' });

    } else if (msg.type === 'transcribe') {
      const out = await asr(msg.audio, { chunk_length_s: 30 });
      self.postMessage({ type: 'transcript', text: (out.text || '').trim() });
    }
  } catch (err) {
    self.postMessage({ type: 'error', op: msg.type, error: String(err?.message || err) });
  }
};
