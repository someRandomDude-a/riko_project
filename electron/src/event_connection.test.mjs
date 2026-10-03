import assert from 'node:assert/strict';
import {test} from 'node:test';
import {connectEvents,createEventHub} from './event_connection.mjs';

function harness() {
  const sockets = [], timers = new Map(), events = [], connections = [];
  class Socket {
    constructor() {sockets.push(this);}
    close() {this.closed = true; this.onclose?.();}
  }
  const stop = connectEvents(event => events.push(event), value => connections.push(value), {
    Socket,
    schedule: callback => {const id = Symbol(); timers.set(id, callback); return id;},
    unschedule: id => timers.delete(id),
  });
  return {sockets, timers, events, connections, stop};
}

test('cleanup closes the reconnected socket, not just the first one', () => {
  const h = harness();
  h.sockets[0].onclose();
  const retry = [...h.timers.values()][0]; h.timers.clear(); retry();
  assert.equal(h.sockets.length, 2);
  h.stop();
  assert.equal(h.sockets[1].closed, true);
  assert.equal(h.timers.size, 0);
});

test('cleanup cancels a pending reconnect and ignores late messages', () => {
  const h = harness(), late = h.sockets[0].onmessage;
  h.sockets[0].onclose();
  h.stop();
  late({data: '{"type":"chat.delta"}'});
  assert.equal(h.timers.size, 0);
  assert.deepEqual(h.events, []);
});

test('malformed events do not break later valid events', () => {
  const h = harness();
  h.sockets[0].onopen();
  for (const data of ['invalid', 'null', '{}', '{"type":"state.snapshot","payload":{}}']) h.sockets[0].onmessage({data});
  assert.deepEqual(h.connections, [true]);
  assert.equal(h.events.length, 1);
  h.stop();
});

test('subscribers share a connection, replay resource state, and release independently',()=>{
  let receiver,connected,opens=0,closes=0;
  const hub=createEventHub((events,status)=>{opens++;receiver=events;connected=status;return()=>{closes++;};});
  const first=[],second=[];
  const stopFirst=hub(e=>first.push(e));connected(true);
  receiver({type:'resource.snapshot',payload:{approvals:{pending:[]}}});
  receiver({type:'resource.approvals',payload:{pending:[{id:'one'}]}});
  const stopSecond=hub(e=>second.push(e));
  assert.equal(opens,1);assert.equal(second.at(-1).payload.pending[0].id,'one');
  stopFirst();assert.equal(closes,0);
  receiver({type:'resource.snapshot',payload:{approvals:{pending:[]}}});
  const third=[];const stopThird=hub(e=>third.push(e));
  assert.equal(third.length,1);assert.deepEqual(third[0].payload.approvals.pending,[]);
  stopSecond();stopThird();assert.equal(closes,1);
});
