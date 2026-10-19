import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { LIMITS } from './core.js';
const require = createRequire(import.meta.url);
const { Miniflare, convertV4MiniflareOptions } = createRequire(require.resolve('wrangler/package.json'))('miniflare');

test('real SQLite Durable Object serializes concurrent spend and preserves it across restarts', async () => {
  const storage = await mkdtemp(path.join(tmpdir(), 'opguard-quota-test-'));
  const modules = [{ type:'ESModule', path:'test.js', contents: `
    import { DemoQuota } from './worker.js';
    export { DemoQuota };
    export default { async fetch(request, env) {
      const body = await request.json();
      const quota = env.DEMO_QUOTA.getByName('groq-demo-v1');
      return Response.json(body.action === 'settle'
        ? await quota.settle(body.id, body.usage) ?? {}
        : await quota.reserve(body.ip, body.budget));
    }};` }];
  for (const file of ['worker.js','core.js','packets.js']) modules.push({type:'ESModule',path:file,contents:await readFile(new URL(file,import.meta.url),'utf8')});
  const options = convertV4MiniflareOptions({name:'quota-test', modules, compatibilityDate:'2026-10-19', durableObjects:{DEMO_QUOTA:{className:'DemoQuota',useSQLite:true}}});
  options.resourcePersistencePath = storage;
  let mf;
  const call = async body => (await mf.dispatchFetch('http://quota.test/', {method:'POST',body:JSON.stringify(body)})).json();
  try {
    mf = new Miniflare(options);
    // Simultaneous calls contend for $2: exactly twenty $0.10 reservations fit.
    const results = await Promise.all(Array.from({length:50}, (_,i) => call({ip:'visitor-'+i,budget:{tokens:1,costNanos:100_000_000}})));
    const accepted = results.filter(result => result.id);
    assert.equal(accepted.length,20);
    assert.equal(results.filter(result => result.reason === 'budget').length,30);
    assert.equal(accepted.length * 100_000_000, LIMITS.dayCostNanos);
    await mf.dispose();
    mf = new Miniflare(options);
    assert.equal((await call({ip:'after-restart',budget:{tokens:1,costNanos:1}})).reason,'budget');
    // Only known provider usage releases a reservation; missing usage releases none.
    await call({action:'settle',id:accepted[0].id});
    assert.equal((await call({ip:'unknown-usage',budget:{tokens:1,costNanos:1}})).reason,'budget');
    await call({action:'settle',id:accepted[0].id,usage:{prompt_tokens:100,completion_tokens:100}});
    assert.ok((await call({ip:'after-settle',budget:{tokens:1,costNanos:99_962_500}})).id);
    assert.equal((await call({ip:'over-again',budget:{tokens:1,costNanos:1}})).reason,'budget');
  } finally {
    await mf?.dispose();
    await rm(storage,{recursive:true,force:true});
  }
});
