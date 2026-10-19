import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {cachedAnswers} from './cached-answers.js';
const source=readFileSync(new URL('../js/rootcause.js',import.meta.url),'utf8');
const bankScript=readFileSync(new URL('../js/rootcause-answers.js',import.meta.url),'utf8');
const canonical=JSON.parse(readFileSync(new URL('./answer-bank.json',import.meta.url),'utf8'));
function run({arm='opguard',search='',bank=cachedAnswers,random=()=>0}={}){
 const links=[],attrs={};let networkCalls=0;
 const createElement=tag=>{
  const element={textContent:'',append(value){this.textContent+=typeof value==='string'?value:value.textContent;}};
  if(tag==='a')links.push(element);return element;
 };
 const result={hidden:true,textContent:'',dataset:{},replaceChildren(){this.textContent='';},append(value){this.textContent+=typeof value==='string'?value:value.textContent;},prepend(value){this.textContent=value.textContent+'\n'+this.textContent;}};
 const button={dataset:{rootcause:arm},addEventListener(_,fn){this.click=fn;},setAttribute(k,v){attrs[k]=v;}};
 const module={querySelectorAll(){return[button];},querySelector(selector){return selector===`[data-rootcause="${arm}"]`?button:null;}};
 vm.runInNewContext(source,{document:{querySelector(){return module;},getElementById(){return result;},createTextNode(t){return t;},createElement},location:{hostname:'',search},window:{OPGUARD_DEMO_ANSWERS:bank,OPGUARD_DEMO_API:'https://must-not-call.test'},fetch(){networkCalls++;throw new Error('No network allowed');},URLSearchParams,Math:{random,floor:Math.floor}});
 return{result,button,links,attrs,get networkCalls(){return networkCalls;}};
}
test('both arms work offline, show answers without the repeated notice, and avoid immediate repeats',()=>{
 for(const arm of ['trace','opguard']){
  const x=run({arm});const seen=new Set();let previous;
  for(let i=0;i<12;i++){
   x.button.click();assert.equal(x.result.hidden,false);assert.doesNotMatch(x.result.textContent,/Reviewed example.*no live model call/);
   assert.notEqual(x.result.dataset.answerId,previous);previous=x.result.dataset.answerId;seen.add(previous);
   assert(cachedAnswers.arms[arm].some(a=>a.id===previous));
  }
  assert(seen.size>=2);assert.equal(x.networkCalls,0);assert.equal(x.button.textContent,'Show another example');assert.match(x.attrs['aria-label'],/another/);
 }
});
test('random selection can reach every variant and the requested deep link shows an answer',()=>{
 for(let i=0;i<3;i++){
  const x=run({search:'?analyze=opguard',random:()=>i/3+.01});
  assert.equal(x.result.dataset.answerId,cachedAnswers.arms.opguard[i].id);assert.equal(x.networkCalls,0);
 }
 assert.equal(run({search:'?analyze=unknown'}).result.hidden,true);
});
test('missing answer bank fails without falling back to a paid API',()=>{
 const x=run({bank:null});x.button.click();assert.match(x.result.textContent,/could not be loaded/);assert.equal(x.networkCalls,0);
});
test('safe formatting and source citations allow only captured local files and lines',()=>{
 const answer='<script>literal</script> **run2/deepspeed/runtime/zero/stage_1_and_2.py:998** `run1/deepspeed/runtime/zero/stage_1_and_2.py:996` run1/deepspeed/runtime/zero/stage_1_and_2.py:9999 javascript:alert(1)';
 const x=run({bank:{notice:'Reviewed example',arms:{opguard:[{id:'safe',answer}]}}});x.button.click();
 assert.match(x.result.textContent,/<script>literal<\/script>/);assert.deepEqual(x.links.map(a=>a.href),['demo/sources/run2-stage_1_and_2.html#L998','demo/sources/run1-stage_1_and_2.html#L996']);assert(x.links.every(a=>a.target==='_blank'&&a.rel==='noopener'));
});
test('all shipped answers match the reviewed bank and citations match captured source',()=>{
 const browser={window:{}};vm.runInNewContext(bankScript,browser);
 const expected={version:canonical.version,notice:canonical.notice,provenance:canonical.provenance,arms:canonical.arms};
 assert.deepEqual(JSON.parse(JSON.stringify(browser.window.OPGUARD_DEMO_ANSWERS)),expected);assert.deepEqual(cachedAnswers,expected);
 const sources={};
 for(const[file,sha]of Object.entries(canonical.source_sha256)){
  const run=file.split('/')[0];const bytes=readFileSync(new URL(`./sources/${run}-stage_1_and_2.py`,import.meta.url));
  assert.equal(createHash('sha256').update(bytes).digest('hex'),sha);sources[file]=bytes.toString().split(/\r?\n/);
 }
 const checked=new Set();
 for(const ref of canonical.source_checks){assert(sources[ref.file][ref.line-1].includes(ref.contains));checked.add(`${ref.file}:${ref.line}`);}
 for(const answer of canonical.arms.opguard)for(const[cite]of answer.answer.matchAll(/run[12]\/deepspeed\/runtime\/zero\/stage_1_and_2\.py:\d+/g))assert(checked.has(cite));
 for(const answer of canonical.arms.trace)assert.doesNotMatch(answer.answer,/stage_1_and_2\.py:\d+/);
});
