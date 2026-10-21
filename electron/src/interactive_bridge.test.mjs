import test from 'node:test';
import assert from 'node:assert/strict';
import {changeReporter} from './interactive_bridge.mjs';

test('a hit test reports to main only when its answer changes, and again after a reset',()=>{
 const sent=[],report=changeReporter(value=>sent.push(value));
 for(const value of [true,1,'yes',false,0,null,undefined,true])report(value);
 assert.deepEqual(sent,[true,false,true],'a moving cursor over the same thing sends nothing more');
 report.reset();report(true);assert.deepEqual(sent,[true,false,true,true],'main changed it (a mode switch): the next report goes out');
});
