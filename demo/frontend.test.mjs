import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
const source = readFileSync(new URL('../js/rootcause.js', import.meta.url), 'utf8');
async function run(events) {
 const history = [];
 const result = {hidden:true, replaceChildren(){this.textContent='';}, append(value){this.textContent += typeof value === 'string' ? value : value.textContent;}, setAttribute(){}, set textContent(v){this.value=v;history.push(v);}, get textContent(){return this.value;}};
 const button = {dataset:{rootcause:'trace'}, addEventListener(_,fn){this.click=fn;}};
 const module = {querySelectorAll(){return [button];}};
 const stream = new ReadableStream({start(c){const bytes=new TextEncoder().encode(events.map(e=>JSON.stringify(e)+'\n').join('')); for(let i=0;i<bytes.length;i+=7)c.enqueue(bytes.slice(i,i+7));c.close();}});
 vm.runInNewContext(source,{document:{querySelector(){return module;},getElementById(){return result;},createTextNode(t){return t;},createElement(){return {textContent:''};}},location:{hostname:'orderlab.io',search:''},window:{OPGUARD_DEMO_API:'https://demo.test/api/rootcause'},fetch:async()=>new Response(stream,{headers:{'content-type':'application/x-ndjson'}}),AbortSignal,URLSearchParams,TextDecoder});
 await button.click(); return {result,button,history};
}
test('streaming UI handles fragmented records and renders the final answer safely',async()=>{
 const x=await run([{progress:'Reading part 1/8…'},{progress:'Waiting 60s…'},{answer:'Observed **difference**; <script>literal</script>'}]);
 assert.ok(x.history.includes('Waiting 60s…')); assert.match(x.result.textContent, /<script>literal<\/script>/);assert.equal(x.button.disabled,false);
});
test('stream failures and missing final answers are visible, never shown as success',async()=>{
 assert.match((await run([{error:'Quota reached',status:429,retryAfter:60}])).result.textContent,/Quota reached/);
 assert.match((await run([{progress:'Reading…'}])).result.textContent,/Could not reach/);
});
