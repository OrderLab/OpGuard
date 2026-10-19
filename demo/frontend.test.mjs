import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
const source = readFileSync(new URL('../js/rootcause.js', import.meta.url), 'utf8');
async function run(events, jsonResponse = null, arm = 'trace') {
 const history = [];
 const links = [];
 const createElement = tag => {
   const element = {textContent:'', append(value){this.textContent += typeof value === 'string' ? value : value.textContent;}};
   if (tag === 'a') links.push(element);
   return element;
 };
 let calls = 0;
 const result = {hidden:true, replaceChildren(){this.textContent='';}, append(value){this.textContent += typeof value === 'string' ? value : value.textContent;}, setAttribute(){}, set textContent(v){this.value=v;history.push(v);}, get textContent(){return this.value;}};
 const button = {dataset:{rootcause:arm}, addEventListener(_,fn){this.click=fn;}};
 const module = {querySelectorAll(){return [button];}};
 const stream = new ReadableStream({start(c){const bytes=new TextEncoder().encode(events.map(e=>JSON.stringify(e)+'\n').join('')); for(let i=0;i<bytes.length;i+=7)c.enqueue(bytes.slice(i,i+7));c.close();}});
 vm.runInNewContext(source,{document:{querySelector(){return module;},getElementById(){return result;},createTextNode(t){return t;},createElement},location:{hostname:'orderlab.io',search:''},window:{OPGUARD_DEMO_API:'https://demo.test/api/rootcause'},fetch:async()=>{calls++; return jsonResponse ? Response.json(jsonResponse.body,{status:jsonResponse.status}) : new Response(stream,{headers:{'content-type':'application/x-ndjson'}});},AbortSignal,URLSearchParams,TextDecoder});
 await button.click(); return {result,button,history,links,get calls(){return calls;}};
}
test('streaming UI handles fragmented records and renders the final answer safely',async()=>{
 const x=await run([{progress:'Reading part 1/8…'},{progress:'Waiting 60s…'},{answer:'Observed **difference**; <script>literal</script>'}]);
 assert.ok(x.history.includes('Waiting 60s…')); assert.match(x.result.textContent, /<script>literal<\/script>/);assert.equal(x.button.disabled,false);
});
test('stream failures and missing final answers are visible, never shown as success',async()=>{
 assert.match((await run([{error:'Quota reached',status:429,retryAfter:60}])).result.textContent,/Quota reached/);
 assert.match((await run([{progress:'Reading…'}])).result.textContent,/Could not reach/);
});
test('one-call JSON answers render safely and restore the button',async()=>{
 const x=await run([],{status:200,body:{answer:'Observed **difference**; <script>literal</script>',partsRead:1}});
 assert.match(x.result.textContent, /<script>literal<\/script>/);
 assert.equal(x.button.disabled,false);
 assert.equal(x.calls,1);
});
test('budget rejection is visible and cooldown prevents another paid request',async()=>{
 const x=await run([],{status:429,body:{error:'The demo has reached its $2 budget for the last 24 hours.',retryAfter:3600}});
 assert.match(x.result.textContent,/\$2.*24 hours/);
 await x.button.click();
 assert.equal(x.calls,1);
 assert.match(x.result.textContent,/try again/i);
});
test('source citations link only to known local captured files and valid line numbers',async()=>{
 const x=await run([],{status:200,body:{answer:'Observed `run2/deepspeed/runtime/zero/stage_1_and_2.py:1736`; **run1/deepspeed/runtime/zero/stage_1_and_2.py:1734**. Invalid run1/deepspeed/runtime/zero/stage_1_and_2.py:9999 and javascript:alert(1).'}});
 assert.deepEqual(x.links.map(a=>a.href),['demo/sources/run2-stage_1_and_2.html#L1736','demo/sources/run1-stage_1_and_2.html#L1734']);
 assert.ok(x.links.every(a=>a.target==='_blank' && a.rel==='noopener'));
 assert.match(x.result.textContent,/javascript:alert\(1\)/);
});
test('OpGuard displays captured locations even when the model omits file citations',async()=>{
 const payload = {status:200,body:{answer:'Observed at line 1734.',sourceLocations:[{rank:0,source_file:'run1/deepspeed/runtime/zero/stage_1_and_2.py',line:1734}]}};
 const x=await run([],payload,'opguard');
 assert.match(x.result.textContent,/Captured callsites \(from the trace\)/);
 assert.deepEqual(x.links.map(a=>a.href),['demo/sources/run1-stage_1_and_2.html#L1734']);
 const baseline=await run([],payload);
 assert.equal(baseline.links.length,0);
 assert.doesNotMatch(baseline.result.textContent,/Captured callsites/);
});
