import assert from 'node:assert/strict';
const endpoint=process.argv[2];
if(!endpoint?.startsWith('https://'))throw new Error('Pass the deployed HTTPS /api/rootcause URL.');
const checks=[
 {method:'GET',status:405},
 {method:'OPTIONS',origin:'https://orderlab.io',status:204},
 {method:'POST',origin:'https://example.com',body:{arm:'trace'},status:403},
 {method:'POST',origin:'https://orderlab.io',body:{arm:'trace',prompt:'custom'},status:400},
 ...['trace','opguard'].map(arm=>({method:'POST',origin:'https://orderlab.io',body:{arm},status:200})),
];
for(const check of checks){
 const r=await fetch(endpoint,{method:check.method,headers:{...(check.origin?{Origin:check.origin}:{}),'Content-Type':'application/json'},...(check.body?{body:JSON.stringify(check.body)}:{}),signal:AbortSignal.timeout(10000)});
 const text=await r.text();assert.equal(r.status,check.status,text);
 assert.equal(r.headers.get('Access-Control-Allow-Origin'),check.origin==='https://orderlab.io'?check.origin:null);
 if(check.status===200){const body=JSON.parse(text);assert.equal(body.mode,'cached');assert.equal(body.inferenceCalls,0);assert.equal(body.model,undefined);assert.ok(body.answerId);assert.match(body.answer,/Reviewed example/);assert.ok(body.answer.length>100);}
 console.log(`${check.method} ${check.body?.arm||''} ${r.status}: passed`);
}
console.log('Cached public API verified; no model calls requested.');
