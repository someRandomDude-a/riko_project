import test from 'node:test';
import assert from 'node:assert/strict';
import {avatarModelURL,validateAvatarFormat} from './avatar_model.mjs';

test('model/format selection changes URL without exposing arbitrary asset paths',()=>{
  const first=avatarModelURL('http://localhost','character_files/models/a.vrm');
  const second=avatarModelURL('http://localhost','character_files/models/b.vrm');
  assert.notEqual(first,second);assert.notEqual(first,avatarModelURL('http://localhost','character_files/models/a.vrm','vrm1'));
  assert.equal(new URL(first).pathname,'/api/avatar/model');
});

test('format settings detect VRM versions and reject mismatches rather than converting',()=>{
  assert.equal(validateAvatarFormat('0'),'vrm0');assert.equal(validateAvatarFormat('1','vrm1'),'vrm1');
  assert.throws(()=>validateAvatarFormat('0','vrm1'),/does not match/);
  assert.throws(()=>validateAvatarFormat('2'),/Unsupported/);
});
