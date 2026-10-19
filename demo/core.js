import { cachedAnswers } from './cached-answers.js';
import { DIAGNOSTIC_PROMPT } from './diagnostic-guide.js';
import { packets, capturedLocations } from './packets.js';

export const MODEL = 'openai/gpt-oss-20b';
export const PROMPT = DIAGNOSTIC_PROMPT;
export const MAX_COMPLETION = 3072;
export const CONTEXT_TOKENS = 131072;
export const CALL_TIMEOUT_MS = 60000;
export const DAY_MS = 86400000;
// Integer nano-USD: $0.075/M input and $0.30/M output, as of 2026-10-19.
// Ignore cache discounts when reserving and settling, keeping spend conservative.
export const PRICING = { input: 75, output: 300 };
export const LIMITS = { dayCostNanos: 2_000_000_000, minuteTokens: 240000, minuteRequests: 120, ipMinuteRequests: 6, ipDayRequests: 200 };
const encoder = new TextEncoder();

export function requestBudget(arm) {
  // Each UTF-8 byte is an upper bound of one input token. Include ample framing
  // overhead instead of depending on the provider matching a local tokenizer.
  // Accepted input cannot exceed the model's context window; reserve the full
  // output allowance on top, conservatively, even though it shares that window.
  const inputTokens = Math.min(encoder.encode(PROMPT + packets[arm]).length + 1024, CONTEXT_TOKENS);
  return { tokens: inputTokens + MAX_COMPLETION, costNanos: inputTokens * PRICING.input + MAX_COMPLETION * PRICING.output };
}

function chargeTime(entry) {
  // Unknown/failed requests keep their full reservation through the latest
  // possible completion time. Successful requests use the settlement time.
  return entry.chargedAt ?? entry.time + CALL_TIMEOUT_MS;
}
function reservedCost(entry) {
  // Preserve pre-upgrade entries in the same Durable Object; never reset spend.
  return entry.costNanos ?? entry.tokens * PRICING.output;
}
function waitForCapacity(entries, now, window, limit, needed, value, timestamp = entry => entry.time) {
  const active = entries.filter(entry => now - timestamp(entry) < window).sort((a, b) => timestamp(a) - timestamp(b));
  let used = active.reduce((sum, entry) => sum + value(entry), 0);
  if (used + needed <= limit) return 0;
  for (const entry of active) {
    used -= value(entry);
    if (used + needed <= limit) return Math.max(1, Math.ceil((timestamp(entry) + window - now) / 1000));
  }
  return Math.ceil(window / 1000);
}

// Adapters MUST run this inside an atomic persistent transaction before fetch.
export function reserve(state, ip, budget, now, id) {
  const entries = (state || []).filter(entry => now - chargeTime(entry) < DAY_MS);
  if (!Number.isSafeInteger(budget?.tokens) || budget.tokens < 1 || budget.tokens > LIMITS.minuteTokens || !Number.isSafeInteger(budget.costNanos) || budget.costNanos < 1 || budget.costNanos > LIMITS.dayCostNanos) {
    return { entries, retry: 86400, reason: 'budget' };
  }
  const perIp = entries.filter(entry => entry.ip === ip);
  const budgetRetry = waitForCapacity(entries, now, DAY_MS, LIMITS.dayCostNanos, budget.costNanos, reservedCost, chargeTime);
  const retry = Math.max(
    budgetRetry,
    waitForCapacity(entries, now, 60000, LIMITS.minuteTokens, budget.tokens, entry => entry.tokens),
    waitForCapacity(entries, now, 60000, LIMITS.minuteRequests, 1, () => 1),
    waitForCapacity(perIp, now, 60000, LIMITS.ipMinuteRequests, 1, () => 1),
    waitForCapacity(perIp, now, DAY_MS, LIMITS.ipDayRequests, 1, () => 1),
  );
  if (retry) return { entries, retry, reason: budgetRetry ? 'budget' : 'rate' };
  return { entries: [...entries, { id, ip, ...budget, time: now }], id };
}

export function settle(entries, id, usage, now = Date.now()) {
  // Never release reservations on failures, timeouts, or incomplete accounting.
  const input = usage?.prompt_tokens;
  const output = usage?.completion_tokens;
  if (!Number.isSafeInteger(input) || input < 1 || !Number.isSafeInteger(output) || output < 0 || !Number.isSafeInteger(input + output)) return entries;
  const pricing = entries.find(entry => entry.id === id)?.pricing ?? PRICING;
  const costNanos = input * pricing.input + output * pricing.output;
  if (!Number.isSafeInteger(costNanos)) return entries;
  return entries.map(entry => entry.id === id ? { ...entry, tokens: input + output, costNanos, chargedAt: now } : entry);
}

export async function handle(request, env, limiter, fetcher = fetch) {
  const origin = request.headers.get('origin');
  const allowed = (env.ALLOWED_ORIGINS || 'https://orderlab.io').split(',').map(value => value.trim());
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
  // Cached is the default. Only an explicit server setting can enable paid calls.
  if (env.DEMO_MODE !== 'live') {
    const answers = cachedAnswers.arms[body.arm];
    const chosen = answers[Math.floor(Math.random() * answers.length)];
    return reply(200, {
      mode: 'cached', answerId: chosen.id, inferenceCalls: 0,
      answer: cachedAnswers.notice + '\n\n' + chosen.answer,
      ...(body.arm === 'opguard' ? { sourceLocations: capturedLocations } : {}),
    });
  }
  if (!env.GROQ_API_KEY) return reply(503, { error: 'Live analysis is not configured yet.' });
  if (request.signal.aborted) return reply(499, { error: 'Analysis cancelled.' });
  const reservation = await limiter.reserve(requestBudget(body.arm));
  if (reservation.retry) return reply(429, {
    error: reservation.reason === 'budget' ? 'The demo has reached its $2 budget for the last 24 hours. Please try again later.' : 'The demo is busy. Please try again shortly.',
    retryAfter: reservation.retry,
  }, { 'Retry-After': String(reservation.retry) });
  let upstream;
  try {
    upstream = await fetcher('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: { Authorization: `Bearer ${env.GROQ_API_KEY}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: MODEL, temperature: 0, reasoning_effort: 'low', include_reasoning: false, max_completion_tokens: MAX_COMPLETION, messages: [{ role: 'system', content: PROMPT }, { role: 'user', content: packets[body.arm] }] }),
      signal: AbortSignal.any([request.signal, AbortSignal.timeout(CALL_TIMEOUT_MS)]),
    });
  } catch { return reply(502, { error: 'The model did not respond. Please try again later.' }); }
  if (upstream.status === 429) {
    const delay = Number(upstream.headers.get('retry-after'));
    const retry = Number.isFinite(delay) && delay > 0 ? Math.min(86400, Math.ceil(delay)) : 60;
    return reply(429, { error: 'The model is busy. Please try again shortly.', retryAfter: retry }, { 'Retry-After': String(retry) });
  }
  if (!upstream.ok) return reply(502, { error: 'Live analysis is temporarily unavailable.' });
  let result;
  try { result = await upstream.json(); } catch { return reply(502, { error: 'The model returned an invalid response.' }); }
  await limiter.settle(reservation.id, result.usage);
  const answer = result.choices?.[0]?.message?.content?.trim();
  if (!answer || result.choices[0].finish_reason === 'length') return reply(502, { error: 'The model did not finish an answer within the demo budget. Please try again.' });
  return reply(200, { answer, model: MODEL, partsRead: 1, ...(body.arm === 'opguard' ? { sourceLocations: capturedLocations } : {}) });
}
