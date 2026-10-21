import test from 'node:test';
import assert from 'node:assert/strict';
// What preload exposes (electron/preload.cjs); api.mjs reads it once, as a window's first script does.
globalThis.rikoConfig={backend:'http://127.0.0.1:9123'};
const {API,EVENTS,socketURL,mediaURL,request}=await import('./api.mjs');

const calls=[];
function reply(...responses){calls.length=0;globalThis.fetch=async(url,init)=>{calls.push([url,init]);const [status,body]=responses.shift();return new Response(body,{status});};}

test('every URL and socket comes from the origin preload handed the page',()=>{
  assert.equal(API,'http://127.0.0.1:9123');assert.equal(EVENTS,'ws://127.0.0.1:9123/ws/events');
  assert.equal(socketURL('/ws/resources/gpu'),'ws://127.0.0.1:9123/ws/resources/gpu');
  assert.equal(mediaURL('a b.png'),'http://127.0.0.1:9123/api/media?path=a%20b.png');
});

test('JSON bodies go out as JSON and come back parsed; an empty reply is null',async()=>{
  reply([200,'{"ok":true}']);
  assert.deepEqual(await request('/api/tasks',{method:'POST',body:{title:'x'}}),{ok:true});
  assert.deepEqual(calls,[['http://127.0.0.1:9123/api/tasks',{method:'POST',headers:{'Content-Type':'application/json'},body:'{"title":"x"}'}]]);
  reply([200,'']);assert.equal(await request('/api/chat/stop',{method:'POST'}),null);
  assert.deepEqual(calls[0][1],{method:'POST',headers:{}});
});

test('a binary upload goes out as is, with its own type',async()=>{
  const png=new Uint8Array([137,80,78,71]);reply([200,'{}']);
  await request('/api/whiteboard/image?revision=3',{method:'POST',type:'image/png',body:png});
  assert.equal(calls[0][1].headers['Content-Type'],'image/png');assert.equal(calls[0][1].body,png);
});

test('failures carry the server reason and status, whatever the error body is',async()=>{
  for(const [status,body,message] of [[409,'{"detail":"Riko is already handling another turn"}','Riko is already handling another turn'],
    [400,'{"detail":{"detail":"Nested reason"}}','Nested reason'],[502,'<html>Bad gateway</html>','<html>Bad gateway</html>'],
    [422,'{"detail":[{"msg":"field required"}]}','{"detail":[{"msg":"field required"}]}'],[500,'','Request failed (500)']]){
    reply([status,body]);
    await assert.rejects(request('/api/tasks'),error=>error.message===message&&error.status===status,String(status));
  }
  reply([404,'{"detail":"Not Found"}']);await assert.rejects(request('/api/discord/status'),/does not support the current Discord controls/);
  globalThis.fetch=async()=>{throw new TypeError('Failed to fetch');};
  await assert.rejects(request('/api/tasks'),error=>error instanceof TypeError&&error.status===undefined);  // no reply at all
});

test('a security-sensitive change is retried once, signed, only if the user allows it',async()=>{
  const challenge='{"action":"settings"}';
  globalThis.riko={confirmChange:async text=>text===challenge?'f'.repeat(64):null};
  reply([428,JSON.stringify({detail:{confirm:challenge}})],[200,'{"saved":true}']);
  assert.deepEqual(await request('/api/settings',{method:'PUT',body:{a:1}}),{saved:true});
  assert.equal(calls.length,2);assert.equal(calls[1][1].headers['X-Riko-Confirmation'],'f'.repeat(64));assert.equal(calls[1][1].body,'{"a":1}');
  globalThis.riko={confirmChange:async()=>null};reply([428,JSON.stringify({detail:{confirm:challenge}})]);
  await assert.rejects(request('/api/settings',{method:'PUT',body:{a:1}}),/Change cancelled/);assert.equal(calls.length,1);
  delete globalThis.riko;
});
