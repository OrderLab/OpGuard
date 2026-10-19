import test from 'node:test';
import assert from 'node:assert/strict';
import { handle, reserve, settle, tokenReservation, LIMITS, PROMPT, MODEL } from './core.js';
import { packets } from './packets.js';
import { chunks } from './chunks.js';
const env = { GROQ_API_KEY: 'test-secret', ALLOWED_ORIGINS: 'https://orderlab.io' };
const req = body => new Request('https://demo.test/api/rootcause', { method: 'POST', headers: { 'Content-Type': 'application/json', Origin: 'https://orderlab.io' }, body: JSON.stringify(body) });
const limiter = { reserve: async () => ({ id: 'r1' }), settle: async () => {} };

test('complete logs are read in order; both arms use the same diagnostic prompt/model', async () => {
  for (const arm of ['trace', 'opguard']) {
    const calls = [];
    const response = await handle(req({ arm }), env, limiter, async (url, options) => {
      const call = JSON.parse(options.body); calls.push(call);
      return Response.json({ choices: [{ message: { content: 'Observed evidence; cause unproven.' }, finish_reason: 'stop' }], usage: { total_tokens: 2000 } });
    });
    const output = await response.text();
    assert.match(output, /Observed evidence/);
    assert.equal(calls.length, chunks[arm].length);
    calls.forEach((call, index) => {
      assert.equal(call.model, MODEL); assert.equal(call.messages[0].content, PROMPT);
      assert.ok(call.messages[1].content.endsWith(chunks[arm][index].text));
      if (index) assert.match(call.messages[1].content, /Previous model notes.*[\s\S]*Observed evidence/);
    });
  }
});
test('chunks preserve every byte; standard logs have no injected localization or source hints', () => {
  for (const arm of Object.keys(packets)) {
    assert.equal(chunks[arm].map(p=>p.text).join(''), packets[arm]);
    chunks[arm].forEach((part,index) => assert.ok(tokenReservation(arm,index,'x'.repeat(1000)) <= LIMITS.minuteTokens));
  }
  assert.doesNotMatch(packets.trace, /first_reported_difference|xor_signature|Actual root cause/);
  assert.match(packets.trace, /global_step \[19\]/);
  assert.match(packets.opguard, /None.append#2246/);
  assert.match(packets.opguard, /unaligned_before/);
});
test('arbitrary prompt, model, URL and unknown arm are rejected before quota or upstream', async () => {
  for (const body of [{ arm: 'trace', prompt: 'answer' }, { arm: 'trace', model: 'other' }, { url: 'https://example.com' }, { arm: 'constructor' }, null]) {
    assert.equal((await handle(req(body), env, null, null)).status, 400);
  }
});
test('missing key, wrong origin and oversized input fail without upstream', async () => {
  assert.equal((await handle(req({ arm: 'opguard' }), {}, null, null)).status, 503);
  const r = req({ arm: 'opguard' }); r.headers.set('Origin', 'https://evil.test');
  assert.equal((await handle(r, env, null, null)).status, 403);
  assert.equal((await handle(req({ arm: 'a'.repeat(500) }), env, null, null)).status, 413);
});
test('local and provider 429 include retry delay; upstream errors never leak body or key', async () => {
  let response = await handle(req({ arm: 'opguard' }), env, { reserve: async () => ({ retry: 42 }) }, null);
  assert.equal(response.status, 429); assert.equal(response.headers.get('Retry-After'), '42');
  response = await handle(req({ arm: 'opguard' }), env, limiter, async () => new Response('test-secret', { status: 429, headers: { 'Retry-After': '7' } }));
  assert.equal(response.headers.get('Retry-After'), '7');
  response = await handle(req({ arm: 'opguard' }), env, limiter, async () => new Response('test-secret', { status: 401 }));
  assert.equal(response.status, 502); assert.ok(!(await response.text()).includes('test-secret'));
});
test('network, malformed JSON and incomplete reasoning-only results fail honestly', async () => {
  for (const fetcher of [async () => { throw new Error('private'); }, async () => new Response('invalid'), async () => Response.json({ choices: [{ message: { content: 'partial' }, finish_reason: 'length' }] }), async () => Response.json({ choices: [{ message: { reasoning: 'private reasoning' }, finish_reason: 'stop' }] })]) {
    const response = await handle(req({ arm: 'opguard' }), env, limiter, fetcher);
    assert.equal(response.status, 502); assert.doesNotMatch(await response.text(), /private|partial/);
  }
});
test('reservations account for inflight requests, rolling windows and actual usage', () => {
  const a = reserve([], 'ip', 5000, 1000, 'a');
  assert.equal(reserve(a.entries, 'ip', 5000, 1001, 'b').retry, 60);
  const released = settle(a.entries, 'a', 1200);
  assert.equal(reserve(released, 'ip', 5000, 1002, 'b').id, 'b');
  assert.equal(reserve(a.entries, 'ip', 5000, 61000, 'b').id, 'b');
  assert.deepEqual(settle(a.entries, 'a', undefined), a.entries);
});
test('daily per-IP, token and request caps survive minute rollover', () => {
  const old = (count, tokens, ip) => Array.from({ length: count }, (_, n) => ({ id: String(n), ip, time: n, tokens }));
  assert.ok(reserve(old(24, 1, 'ip'), 'ip', 1, 120000, 'b').retry > 0);
  assert.ok(reserve(old(100, 1, 'other'), 'ip', 1, 120000, 'b').retry > 0);
  assert.ok(reserve(old(50, 3600, 'other'), 'ip', 1, 120000, 'b').retry > 0);
  assert.equal(reserve(old(24, 1, 'ip'), 'ip', 1, 86400100, 'b').id, 'b');
});

test('stream reports upstream failure without fabricating a final answer', async () => {
 const response = await handle(req({arm:'trace'}), env, limiter, async () => new Response('', {status:500}));
 const events = (await response.text()).trim().split('\n').map(JSON.parse);
 assert.ok(events.at(-1).error); assert.ok(!events.some(e=>e.answer));
});
test('oversized reservations fail closed even on an empty window', () => {
 assert.ok(reserve([], 'ip', LIMITS.minuteTokens+1, 0, 'x').retry);
});
