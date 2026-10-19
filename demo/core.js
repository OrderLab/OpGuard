import { chunks } from './chunks.js';

export const MODEL = 'openai/gpt-oss-20b';
export const PROMPT = 'Diagnose the training divergence using only this evidence. In at most 120 words, separate: observed localization; root-cause hypothesis; missing proof. Cite observations or source lines. Source code is not evidence that a path executed. Export timestamps are presentation fields, not GPU timing. XOR signatures are NOT tensor values: equal XOR does NOT establish equal inputs; differing XOR cannot measure numerical error. If the cause is undetermined, say so.';
export const MAX_COMPLETION = 3072;
export const LIMITS = { minuteTokens: 7800, dayTokens: 180000, dayRequests: 100, ipRequests: 24 };
const encoder = new TextEncoder();

export function tokenReservation(arm, index = 0, notes = '') {
  return chunks[arm][index].tokens + encoder.encode(PROMPT + notes).length + MAX_COMPLETION + 512;
}

// Pure policy; adapters must call this under an atomic, persistent transaction.
export function reserve(state, ip, tokens, now, id) {
  const entries = (state || []).filter(e => now - e.time < 86400000);
  const minute = entries.filter(e => now - e.time < 60000);
  const perIp = entries.filter(e => e.ip === ip);
  let retry = 0;
  if (entries.length >= LIMITS.dayRequests || perIp.length >= LIMITS.ipRequests || entries.reduce((n,e) => n + e.tokens, 0) + tokens > LIMITS.dayTokens) {
    const relevant = perIp.length >= LIMITS.ipRequests ? perIp : entries;
    retry = Math.max(1, Math.ceil((relevant[0].time + 86400000 - now) / 1000));
  } else if (minute.reduce((n,e) => n + e.tokens, 0) + tokens > LIMITS.minuteTokens) {
    retry = Math.max(1, Math.ceil(((minute[0]?.time ?? now) + 60000 - now) / 1000));
  }
  if (tokens > LIMITS.minuteTokens) return { entries, retry: 86400 };
  if (retry) return { entries, retry };
  return { entries: [...entries, { id, ip, tokens, time: now }], id };
}

export function settle(entries, id, usage) {
  // Fail closed if the provider omits usage, or a request fails after reservation.
  if (!Number.isSafeInteger(usage) || usage < 1) return entries;
  return entries.map(e => e.id === id ? { ...e, tokens: Math.min(e.tokens, usage) } : e);
}

export async function handle(request, env, limiter, fetcher = fetch) {
  const origin = request.headers.get('origin');
  const allowed = (env.ALLOWED_ORIGINS || 'https://orderlab.io').split(',').map(x => x.trim());
  const headers = { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', 'Vary': 'Origin' };
  if (origin && allowed.includes(origin)) headers['Access-Control-Allow-Origin'] = origin;
  const reply = (status, data, extra = {}) => new Response(JSON.stringify(data), { status, headers: { ...headers, ...extra } });
  if (origin && !allowed.includes(origin)) return reply(403, { error: 'This origin is not allowed.' });
  if (new URL(request.url).pathname !== '/api/rootcause') return reply(404, { error: 'Not found.' });
  if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: { ...headers, 'Access-Control-Allow-Methods': 'POST', 'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Max-Age': '600' } });
  if (request.method !== 'POST') return reply(405, { error: 'Use POST.' }, { Allow: 'POST' });
  if (!request.headers.get('content-type')?.startsWith('application/json')) return reply(415, { error: 'Expected JSON.' });
  let body;
  try {
    // Bounded reading also protects chunked requests with no Content-Length.
    const reader = request.body?.getReader();
    if (!reader) throw new Error();
    let raw = ''; let size = 0;
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > 128) { await reader.cancel(); return reply(413, { error: 'Request too large.' }); }
      raw += decoder.decode(value, { stream: true });
    }
    body = JSON.parse(raw + decoder.decode());
  } catch { return reply(400, { error: 'Invalid JSON.' }); }
  if (!body || Object.keys(body).length !== 1 || !['trace', 'opguard'].includes(body.arm)) return reply(400, { error: 'Choose trace or opguard.' });
  if (!env.GROQ_API_KEY) return reply(503, { error: 'Live analysis is not configured yet.' });
  const abort = new AbortController();
  const cancel = () => abort.abort();
  request.signal.addEventListener('abort', cancel, { once: true });
  async function call(index, notes) {
  if (encoder.encode(notes).length > 1000) return reply(502, { error: 'Intermediate notes exceeded the reading budget. Please try again.' });
  const reservation = await limiter.reserve(tokenReservation(body.arm, index, notes));
  if (reservation.retry) return reply(429, { error: 'Free demo limit reached. Please try again later.', retryAfter: reservation.retry }, { 'Retry-After': String(reservation.retry) });
  let upstream;
  try {
    upstream = await fetcher('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: { Authorization: `Bearer ${env.GROQ_API_KEY}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: MODEL, temperature: 0, reasoning_effort: 'low', include_reasoning: false, max_completion_tokens: MAX_COMPLETION, messages: [{ role: 'system', content: PROMPT }, { role: 'user', content: (chunks[body.arm].length > 1 ? `Read part ${index + 1}/${chunks[body.arm].length} of the complete logs, in order. ${index + 1 === chunks[body.arm].length ? 'All parts have now been read; give the final diagnosis.' : 'This is not the final part. In at most 80 words, maintain compact evidence notes for the next part, including exact metrics and unresolved clues; do not finalize a diagnosis.'} Previous model notes (not original evidence):\n${notes || '(none)'}\nNext original log part:\n` : '') + chunks[body.arm][index].text }] }),
      signal: AbortSignal.any([abort.signal, AbortSignal.timeout(45000)]),
    });
  } catch { return reply(502, { error: 'The model did not respond. Please try again later.' }); }
  if (upstream.status === 429) {
    const retry = Math.min(86400, Math.max(1, Number(upstream.headers.get('retry-after')) || 60));
    return reply(429, { error: 'Groq’s free quota is busy. Please try again later.', retryAfter: retry }, { 'Retry-After': String(retry) });
  }
  if (!upstream.ok) return reply(502, { error: 'Live analysis is temporarily unavailable.' });
  let result;
  try { result = await upstream.json(); } catch { return reply(502, { error: 'The model returned an invalid response.' }); }
  await limiter.settle(reservation.id, result.usage?.total_tokens);
  const answer = result.choices?.[0]?.message?.content?.trim();
  if (!answer || result.choices[0].finish_reason === 'length') return reply(502, { error: 'The model did not finish an answer within the demo budget. Please try again.' });
  return reply(200, { answer, model: MODEL });
  }
  if (body.arm === 'opguard') {
    try { return await call(0, ''); }
    finally { request.signal.removeEventListener('abort', cancel); }
  }
  const stream = new ReadableStream({
    start(controller) {
      const send = data => { if (!abort.signal.aborted) controller.enqueue(encoder.encode(JSON.stringify(data) + '\n')); };
      (async () => {
        let notes = ''; const deadline = Date.now() + 15 * 60000;
        for (let index = 0; index < chunks.trace.length; index++) {
          while (!abort.signal.aborted) {
            send({ progress: `Reading complete logs: part ${index + 1}/${chunks.trace.length}…` });
            const response = await call(index, notes);
            const data = await response.json();
            if (response.status === 429 && data.retryAfter <= 60 && Date.now() < deadline) {
              let remaining = data.retryAfter;
              while (remaining > 0 && !abort.signal.aborted) {
                send({ progress: `Free quota: waiting ${remaining}s before part ${index + 1}/${chunks.trace.length}…` });
                const seconds = Math.min(10, remaining);
                await new Promise(resolve => {
                  const done = () => { clearTimeout(timer); abort.signal.removeEventListener('abort', done); resolve(); };
                  const timer = setTimeout(done, seconds * 1000);
                  abort.signal.addEventListener('abort', done, { once: true });
                });
                remaining -= seconds;
              }
              continue;
            }
            if (!response.ok) { send({ ...data, status: response.status }); return; }
            notes = data.answer;
            if (index + 1 < chunks.trace.length && encoder.encode(notes).length > 1000) {
              while (encoder.encode(notes).length > 960) notes = notes.slice(0, -1);
              notes += ' [model notes truncated]';
            }
            break;
          }
          if (abort.signal.aborted) return;
        }
        send({ answer: notes, model: MODEL, partsRead: chunks.trace.length });
      })().catch(() => send({ error: 'Log analysis was interrupted. Please try again.' }))
        .finally(() => { request.signal.removeEventListener('abort', cancel); if (!abort.signal.aborted) controller.close(); });
    },
    cancel() { abort.abort(); },
  });
  return new Response(stream, { headers: { ...headers, 'Content-Type': 'application/x-ndjson', 'X-Content-Type-Options': 'nosniff' } });
}
