import test from 'node:test';
import assert from 'node:assert/strict';
import {createWakeAnimationPlayer} from './avatar_animation.mjs';

const flush = async () => {for(let i=0;i<6;i++)await Promise.resolve();};
function setup(load) {
  const reports=[],disposed=[],mixers=[];
  const player=createWakeAnimationPlayer({load,createClip:()=>({}),dispose:asset=>disposed.push(asset),
    report:(...args)=>reports.push(args),createMixer:root=>{
      const playback={play(){this.played=true;}};
      const mixer={clipAction:()=>playback,addEventListener:(_,fn)=>{mixer.finished=fn;},
        update(){},stopAllAction(){this.stopped=true;},uncacheRoot(){},getRoot:()=>root};
      mixers.push(mixer);return mixer;
    }});
  return {player,reports,disposed,mixers};
}
const action=id=>({id,payload:{path:'character_files/wake.vrma'}});
const asset=()=>({userData:{vrmAnimations:[{}]}});
const vrm={scene:{}};

test('wake animation loads once, starts and reports actual completion',async()=>{
  let loads=0;const state=setup(async()=>{loads++;return asset();});
  state.player.update(action('one'),vrm,.032);await flush();
  assert.equal(state.player.playing,true);
  state.player.update(action('one'),vrm,.032);await flush();
  assert.equal(loads,1);
  state.mixers[0].finished();
  assert.deepEqual(state.reports,[['one','started'],['one','completed']]);
  assert.equal(state.player.playing,false);
  assert.equal(state.disposed.length,1);
  state.player.dispose();
});

test('expired animation loads are disposed without replay',async()=>{
  let resolve;const state=setup(()=>new Promise(done=>{resolve=done;}));
  state.player.update(action('old'),vrm,.032);await flush();
  state.player.update(null,vrm,.032);
  resolve(asset());await flush();
  assert.deepEqual(state.reports,[]);assert.equal(state.disposed.length,1);
  assert.equal(state.mixers.length,0);state.player.dispose();
});

test('unmount cancels pending animation loads',async()=>{
  let resolve;const state=setup(()=>new Promise(done=>{resolve=done;}));
  state.player.update(action('old'),vrm,.032);await flush();state.player.dispose();
  resolve(asset());await flush();
  assert.deepEqual(state.reports,[]);assert.equal(state.disposed.length,1);
});

test('invalid assets report errors and release the animation lane',async()=>{
  const state=setup(async()=>({userData:{}}));
  state.player.update(action('invalid'),vrm,.032);await flush();
  assert.equal(state.reports[0][1],'error');assert.equal(state.player.playing,false);
  assert.equal(state.disposed.length,1);state.player.dispose();
});
