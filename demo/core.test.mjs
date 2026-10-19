import test from 'node:test';
import assert from 'node:assert/strict';
import { handle, reserve, settle, requestBudget, LIMITS, PRICING, PROMPT, MODEL, MAX_COMPLETION, DAY_MS, CALL_TIMEOUT_MS } from './core.js';
import { packets } from './packets.js';
import { encode } from 'gpt-tokenizer/encoding/o200k_base';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
const env = { GROQ_API_KEY: 'test-secret', ALLOWED_ORIGINS: 'https://orderlab.io' };
const req = body => new Request('https://demo.test/api/rootcause', { method: 'POST', headers: { 'Content-Type': 'application/json', Origin: 'https://orderlab.io' }, body: JSON.stringify(body) });
const limiter = { reserve: async () => ({ id: 'r1' }), settle: async () => {} };
const budget = (tokens = 1, costNanos = 1) => ({ tokens, costNanos });
const entry = (overrides = {}) => ({ id: 'old', ip: 'other', time: 0, chargedAt: 0, ...budget(), ...overrides });

test('each arm sends its complete original evidence in one call with identical model and prompt', async () => {
  for (const arm of ['trace', 'opguard']) {
    const calls = [];
    const response = await handle(req({ arm }), env, limiter, async (url, options) => {
      const call = JSON.parse(options.body); calls.push(call);
      return Response.json({ choices: [{ message: { content: 'Observed evidence; cause unproven.' }, finish_reason: 'stop' }], usage: { prompt_tokens: 19000, completion_tokens: 500 } });
    });
    assert.equal(response.status, 200);
    assert.equal((await response.json()).partsRead, 1);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].model, MODEL);
    assert.equal(calls[0].messages[0].content, PROMPT);
    assert.equal(calls[0].messages[1].content, packets[arm]);
    assert.equal(calls[0].max_completion_tokens, MAX_COMPLETION);
    assert.equal(calls[0].tools, undefined);
  }
  assert.doesNotMatch(packets.trace, /first_reported_difference|xor_signature|Actual root cause/);
  assert.match(packets.trace, /global_step \[19\]/);
  assert.match(packets.opguard, /None.append#2246/);
  assert.match(packets.opguard, /unaligned_before/);
});
test('fixed evidence fits the model and monetary reservations exceed tokenized maximum cost', () => {
  for (const arm of ['trace', 'opguard']) {
    const input = encode(PROMPT).length + encode(packets[arm]).length + 1024;
    const reservation = requestBudget(arm);
    assert.ok(input + MAX_COMPLETION < 131072);
    assert.ok(reservation.tokens >= input + MAX_COMPLETION);
    assert.ok(reservation.tokens < LIMITS.minuteTokens);
    assert.ok(reservation.costNanos >= input * PRICING.input + MAX_COMPLETION * PRICING.output);
  }
});
test('standard logs get recorded versions; only OpGuard gets captured source and mapped callsites', () => {
  const marker = '\n--- Runtime observations ---\n';
  const shared = packets.trace.split(marker)[0];
  assert.ok(packets.opguard.startsWith(shared));
  const runtime = JSON.parse(readFileSync(new URL('./evidence/manifest.json', import.meta.url))).runtime;
  assert.ok(shared.includes(JSON.stringify(runtime, null, 2)));
  assert.match(shared, /git-hash=unknown/);
  assert.doesNotMatch(shared + PROMPT, /stage_1_and_2|get_grad_norm_direct|all_norms|overlap_comm|Megatron/i);
  assert.doesNotMatch(packets.trace, /OpGuard captured source|complete module|opguard_wrapped__|get_grad_norm_direct|captured_source_line|source_at_observation/);
  const manifest = JSON.parse(readFileSync(new URL('./evidence/source-manifest.json',import.meta.url)));
  for (const file of manifest.sources) {
    const raw = readFileSync(new URL('./evidence/'+file.path,import.meta.url));
    assert.equal(createHash('sha256').update(raw).digest('hex'),file.sha256);
    const lines=raw.toString().replaceAll('overlap-off','run1').replaceAll('overlap-on','run2').split(/\r?\n/);
    if(lines.at(-1)==='') lines.pop();
    assert.ok(packets.opguard.includes(lines.map((line,i)=>`${i+1}:` + (line ? ` ${line}` : '')).join('\n')));
    const page=readFileSync(new URL('./sources/'+file.path.replace('.py','.html'),import.meta.url),'utf8');
    assert.ok(page.includes(`id="L${file.lines}"`));
  }
  for(const packet of Object.values(packets)) {
    assert.doesNotMatch(packet,/overlap-off|overlap-on|run1 overlap_comm=False|run2 overlap_comm=True/);
    assert.match(packet,/Launch commands are not supplied/);
    assert.match(packet,/Do not assume.*only one difference/);
  }
  for (const [side, run] of [['overlap-off', 'run1'], ['overlap-on', 'run2']]) {
    for (const stream of ['stdout', 'stderr']) {
      const log = readFileSync(new URL(`./evidence/${side}-${stream}.log`, import.meta.url), 'utf8')
        .replaceAll('overlap-off', 'run1').replaceAll('overlap-on', 'run2');
      assert.ok(packets.trace.includes(`--- ${run} / ${stream} ---\n` + log));
    }
  }
  const report=JSON.parse(packets.opguard.split(marker)[1]);
  for(const rank of report.ranks) for(const observed of rank.observations) {
    assert.equal(observed.source_at_observation[0].line,observed.captured_source_line);
    assert.ok(observed.source_at_observation[0].text.includes(String(observed.operator_tag_line)));
  }
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
test('budget exhaustion blocks upstream and explains the rolling $2 cap', async () => {
  const response = await handle(req({ arm: 'trace' }), env, { reserve: async () => ({ retry: 42, reason: 'budget' }) }, () => assert.fail('must not call Groq'));
  assert.equal(response.status, 429);
  assert.equal(response.headers.get('Retry-After'), '42');
  assert.match((await response.json()).error, /\$2.*24 hours/);
});
test('local and provider 429 include retry delay; provider failures are never retried or leaked', async () => {
  let response = await handle(req({ arm: 'opguard' }), env, { reserve: async () => ({ retry: 42 }) }, null);
  assert.equal(response.status, 429); assert.equal(response.headers.get('Retry-After'), '42');
  let calls = 0;
  response = await handle(req({ arm: 'trace' }), env, limiter, async () => { calls++; return new Response('test-secret', { status: 429, headers: { 'Retry-After': '7' } }); });
  assert.equal(calls, 1);
  assert.equal(response.status, 429);
  assert.equal(response.headers.get('Retry-After'), '7');
  response = await handle(req({ arm: 'opguard' }), env, limiter, async () => new Response('test-secret', { status: 401 }));
  assert.equal(response.status, 502); assert.ok(!(await response.text()).includes('test-secret'));
});
test('network, malformed JSON and incomplete reasoning-only results fail honestly', async () => {
  for (const fetcher of [async () => { throw new Error('private'); }, async () => new Response('invalid'), async () => Response.json({ choices: [{ message: { content: 'partial' }, finish_reason: 'length' }] }), async () => Response.json({ choices: [{ message: { reasoning: 'private reasoning' }, finish_reason: 'stop' }] })]) {
    const response = await handle(req({ arm: 'trace' }), env, limiter, fetcher);
    assert.equal(response.status, 502); assert.doesNotMatch(await response.text(), /private|partial/);
  }
});
test('minute token capacity includes inflight requests and returns the first sufficient reset', () => {
  const a = reserve([], 'ip', budget(150000), 1000, 'a');
  assert.equal(reserve(a.entries, 'ip', budget(100000), 1001, 'b').retry, 60);
  const released = settle(a.entries, 'a', { prompt_tokens: 1000, completion_tokens: 200 }, 1002);
  assert.equal(reserve(released, 'ip', budget(100000), 1003, 'b').id, 'b');
  assert.equal(reserve(a.entries, 'ip', budget(100000), 61000, 'b').id, 'b');
  const entries = [entry({time:1000,tokens:100000}), entry({time:2000,tokens:100000})];
  assert.equal(reserve(entries, 'ip', budget(150000), 3000, 'b').retry, 59);
});
test('the $2 budget counts concurrent reservations; exact boundary is allowed', () => {
  const a = reserve([], 'a', budget(1, 1_900_000_000), 0, 'a');
  const b = reserve(a.entries, 'b', budget(1, 100_000_000), 1, 'b');
  assert.equal(b.id, 'b');
  const denied = reserve(b.entries, 'c', budget(), 2, 'c');
  assert.equal(denied.reason, 'budget');
  assert.equal(denied.entries.reduce((sum, e) => sum + e.costNanos, 0), LIMITS.dayCostNanos);
  assert.equal(denied.id, undefined);
});
test('settlement counts input and reasoning/output tokens at full rates, without cache discounts', () => {
  const a = reserve([], 'ip', requestBudget('trace'), 0, 'a');
  const settled = settle(a.entries, 'a', { prompt_tokens: 20000, completion_tokens: 3000, prompt_tokens_details: { cached_tokens: 19000 }, completion_tokens_details: { reasoning_tokens: 2800 } }, 1000);
  assert.equal(settled[0].costNanos, 2_400_000);
  assert.equal(settled[0].tokens, 23000);
  for (const usage of [undefined, {total_tokens:20000}, {prompt_tokens:1,completion_tokens:-1}, {prompt_tokens:NaN,completion_tokens:1}]) assert.deepEqual(settle(a.entries, 'a', usage, 2000), a.entries);
  // Accounting anomalies must not be hidden by clamping down to the reservation.
  assert.equal(settle([entry({id:'a'})], 'a', {prompt_tokens:2,completion_tokens:1}, 1000)[0].costNanos, 450);
});
test('late completions and uncertain failures remain charged for a full rolling day', () => {
  const a = reserve([], 'a', budget(1, LIMITS.dayCostNanos), 0, 'a');
  assert.equal(reserve(a.entries, 'b', budget(), DAY_MS + CALL_TIMEOUT_MS - 1, 'b').reason, 'budget');
  assert.equal(reserve(a.entries, 'b', budget(), DAY_MS + CALL_TIMEOUT_MS, 'b').id, 'b');
  const paid = [entry({costNanos:LIMITS.dayCostNanos, chargedAt:50000})];
  assert.equal(reserve(paid, 'b', budget(), DAY_MS, 'b').retry, 50);
  assert.equal(reserve(paid, 'b', budget(), DAY_MS + 50000, 'b').id, 'b');
});
test('legacy quota entries retain a conservative cost across deployment', () => {
  const old = [{id:'old',ip:'other',time:0,tokens:180000}];
  const remaining = LIMITS.dayCostNanos - 180000 * PRICING.output;
  assert.equal(reserve(old, 'new', budget(1,remaining), 100000, 'new').id, 'new');
  assert.equal(reserve(old, 'new', budget(1,remaining + 1), 100000, 'new').reason, 'budget');
});
test('global and per-IP request caps block bursts and survive minute rollover', () => {
  const rows = (count, ip, start=0) => Array.from({length:count}, (_, n) => entry({id:String(n), ip, time:start+n}));
  assert.ok(reserve(rows(LIMITS.minuteRequests,'other'), 'new', budget(), 1000, 'b').retry);
  assert.ok(reserve(rows(LIMITS.ipMinuteRequests,'ip'), 'ip', budget(), 1000, 'b').retry);
  const daily = rows(LIMITS.ipDayRequests,'ip');
  assert.ok(reserve(daily, 'ip', budget(), 120000, 'b').retry);
  assert.equal(reserve(daily, 'another', budget(), 120000, 'b').id, 'b');
  assert.equal(reserve(daily, 'ip', budget(), DAY_MS, 'b').id, 'b');
});
test('invalid or oversized reservations fail closed on an empty ledger', () => {
  for (const b of [undefined, budget(0), budget(-1), budget(Infinity), budget(1, NaN), budget(1, LIMITS.dayCostNanos+1), budget(LIMITS.minuteTokens+1)]) assert.ok(reserve([], 'ip', b, 0, 'x').retry);
});
test('failed model calls keep their full reservation and cannot evade the daily budget', async () => {
  let entries = [entry({costNanos:LIMITS.dayCostNanos-requestBudget('trace').costNanos})];
  const actualLimiter = {
    reserve: async b => { const r = reserve(entries, 'ip', b, 1000, 'failed'); entries = r.entries; return r; },
    settle: async (id, usage) => { entries = settle(entries,id,usage,2000); },
  };
  const failed = await handle(req({arm:'trace'}), env, actualLimiter, async () => { throw new Error('timeout'); });
  assert.equal(failed.status,502);
  const denied = await handle(req({arm:'opguard'}), env, actualLimiter, () => assert.fail('must not call Groq'));
  assert.equal(denied.status,429);
  assert.equal(entries.reduce((sum,e)=>sum+e.costNanos,0), LIMITS.dayCostNanos);
});
