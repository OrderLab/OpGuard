import assert from 'node:assert/strict';

const endpoint = process.argv[2];
if (!endpoint || !endpoint.startsWith('https://')) throw new Error('Pass the deployed HTTPS /api/rootcause URL.');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
// New workers.dev subdomains can need a few minutes to provision TLS.
let ready = false;
for (let attempt = 0; attempt < 30; attempt++) {
  try {
    const response = await fetch(endpoint, { signal: AbortSignal.timeout(10000) });
    assert.equal(response.status, 405);
    ready = true;
    break;
  } catch (error) {
    if (attempt % 3 === 0) console.log('Waiting for public HTTPS:', error.cause?.code || error.message);
    await sleep(10000);
  }
}
assert.ok(ready, 'Public HTTPS is not ready; no model calls were made.');
const checks = [
  { method: 'OPTIONS', origin: 'https://orderlab.io', status: 204 },
  { method: 'POST', origin: 'https://example.com', body: { arm: 'trace' }, status: 403 },
  { method: 'POST', origin: 'https://orderlab.io', body: { arm: 'trace', prompt: 'custom' }, status: 400 },
  ...['trace', 'opguard'].map(arm => ({ method: 'POST', origin: 'https://orderlab.io', body: { arm }, status: 200 })),
];
for (const check of checks) {
  const call = () => fetch(endpoint, {
    method: check.method,
    headers: { Origin: check.origin, 'Content-Type': 'application/json' },
    ...(check.body ? { body: JSON.stringify(check.body) } : {}),
    signal: AbortSignal.timeout(75000),
  });
  let response = await call();
  if (response.status === 429) {
    const seconds = Number(response.headers.get('Retry-After'));
    if (seconds > 0 && seconds <= 60) {
      console.log(`Waiting ${seconds}s for the rolling demo limit.`);
      await sleep(seconds * 1000);
      response = await call();
    }
  }
  let text = '';
  if (response.headers.get('content-type')?.includes('application/x-ndjson')) {
    const reader = response.body.getReader(); const decoder = new TextDecoder(); let pending = '';
    while (true) {
      const {value,done} = await reader.read(); if (done) break;
      const fragment = decoder.decode(value, {stream:true}); text += fragment; pending += fragment;
      let end;
      while ((end = pending.indexOf('\n')) >= 0) {
        const event = JSON.parse(pending.slice(0,end)); pending = pending.slice(end+1);
        if (event.progress?.startsWith('Reading')) console.log(event.progress);
        if (event.error) console.log(event.error);
      }
    }
  } else text = await response.text();
  assert.equal(response.status, check.status, text);
  if (check.origin === 'https://orderlab.io') assert.equal(response.headers.get('Access-Control-Allow-Origin'), check.origin);
  else assert.equal(response.headers.get('Access-Control-Allow-Origin'), null);
  if (check.status === 200) {
    const result = response.headers.get('content-type')?.includes('application/x-ndjson') ? text.trim().split('\n').map(JSON.parse).at(-1) : JSON.parse(text);
    assert.equal(result.partsRead, 1, JSON.stringify(result));
    assert.equal(result.model, 'openai/gpt-oss-20b');
    assert.ok(result.answer?.length > 0);
    if (check.body.arm === 'opguard') {
      assert.equal(result.sourceLocations?.length, 4);
      assert.deepEqual(result.sourceLocations.map(location => location.line), [1734, 1736, 1745, 1747]);
    } else assert.equal(result.sourceLocations, undefined);
    console.log(JSON.stringify({ arm: check.body.arm, ...result }));
  } else console.log(`${check.method} ${check.status}: passed`);
}
console.log('Public deployment checks passed.');
