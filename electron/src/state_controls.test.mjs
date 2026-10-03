import assert from 'node:assert/strict';
import {test} from 'node:test';
import {cameraDistance, resizeHeldAvatar} from './avatar_geometry.mjs';
import {mergeHistory} from './chat_history.mjs';
import {readBoardView, saveBoardView} from './board_view.mjs';

test('camera framing accounts for tall and wide models plus depth',()=>{
  const size={x:2,y:3,z:.5},aspect=.5,distance=cameraDistance(size,aspect);
  assert.ok(2*(distance-size.z/2)*Math.tan(Math.PI/12)>size.y);
  assert.ok(2*(distance-size.z/2)*Math.tan(Math.PI/12)*aspect>size.x);
});
test('scroll resize keeps a held point anchored and stays within display bounds',()=>{
  const rect={x:100,y:100,width:400,height:600};
  const next=resizeHeldAvatar(rect,-100,{x:300,y:400},{width:1920,height:1080});
  assert.ok(next.width>rect.width && next.height>rect.height);
  assert.ok(Math.abs((300-next.x)/next.width-.5)<.01);
  const huge=resizeHeldAvatar(rect,-10000,{x:300,y:400},{width:1920,height:1080});
  assert.ok(huge.height<=1080 && huge.x>=0 && huge.y>=0);
});
test('history pages merge without duplicates or overwriting newer live text',()=>{
  const result=mergeHistory([{id:'live',text:'newer',event_sequence:10}],
    [{id:'old',sequence:1,text:'past'},{id:'live',sequence:2,text:'stale',event_sequence:9}]);
  assert.deepEqual(result.map(m=>m.id),['old','live']);
  assert.equal(result[1].text,'newer');
});
test('whiteboard user view and positions persist independently of model layout',()=>{
  const memory=new Map();globalThis.localStorage={getItem:key=>memory.get(key),setItem:(key,value)=>memory.set(key,value)};
  const value={page:'page-2',view:{x:1,y:2,z:1.5},positions:{object:{x:20,y:30}},seen:'object'};
  saveBoardView('board',value);assert.deepEqual(readBoardView('board'),value);
  memory.set('board','corrupted');assert.equal(readBoardView('board'),null);
  delete globalThis.localStorage;
});
