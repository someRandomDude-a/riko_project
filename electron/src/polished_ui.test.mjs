import test,{before,after} from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {createServer} from 'vite';
import {normalizePreferences} from './ui_preferences.mjs';
let server,Explanation,Context,Appearance,Tabs,dockPosition,DockControls,Message;
before(async()=>{
 server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom'});
 ({default:Explanation,ExplanationContext:Context}=await server.ssrLoadModule('/src/explanation.jsx'));
 ({default:Appearance}=await server.ssrLoadModule('/src/appearance_page.jsx'));
 ({SettingsTabs:Tabs}=await server.ssrLoadModule('/src/settings_navigation.jsx'));
 ({dockPosition}=await server.ssrLoadModule('/src/overlay_dock.jsx'));
 ({default:DockControls}=await server.ssrLoadModule('/src/dock_controls.jsx'));
 ({Message}=await server.ssrLoadModule('/src/stream_chat.jsx'));
});
after(async()=>{await server?.close();});
test('simple explanations lead with guidance and technical notes can be opened by preference',()=>{
 const render=advanced=>renderToStaticMarkup(React.createElement(Context.Provider,{value:advanced},React.createElement(Explanation,{simple:'Use less memory.'},React.createElement('p',null,'Full technical note.'))));
 assert.doesNotMatch(render(false),/<details open/);assert.match(render(true),/<details open/);
 assert.ok(render(false).indexOf('Use less memory.')<render(false).indexOf('Full technical note.'));
});
test('settings tabs expose selected state and one keyboard entry point',()=>{
 const html=renderToStaticMarkup(React.createElement(Tabs,{groups:[['models','Models'],['voice','Microphone']],group:'voice',onSelect:()=>{}}));
 assert.match(html,/role="tablist"/);assert.match(html,/aria-selected="true" tabindex="0"/);assert.match(html,/aria-selected="false" tabindex="-1"/);
});
test('appearance centralizes materials, motion, explanations and speech cloud preferences',()=>{
 const html=renderToStaticMarkup(React.createElement(Appearance,{preferences:normalizePreferences({}),updatePreferences:()=>{}}));
 for(const text of ['Liquid glass','Mini and collapsed backgrounds stay transparent','Glass panel opacity','Glass blur','Reduce motion','Advanced explanations','Assistant speech bubble','Mic button border trigger','Screen border trigger','Dock border thickness','React to microphone audio level'])assert.ok(html.includes(text),text);
 assert.ok(!html.includes('Main window opacity'));assert.ok(!html.includes('Whiteboard opacity'));
});
test('dock positions stay on the current display after dragging or resize',()=>{
 assert.deepEqual(dockPosition(-50,-20,1000,700),{x:8,y:8});
 assert.deepEqual(dockPosition(5000,5000,1000,700),{x:888,y:588});
});
test('collapsed dock has only the microphone and bottom bar; mini chat adds mute and send',()=>{
 const props={preferences:normalizePreferences({}),onExpand:()=>{}};
 const collapsed=renderToStaticMarkup(React.createElement(DockControls,{...props,collapsed:true}));
 assert.equal((collapsed.match(/<button/g)||[]).length,2);
 assert.match(collapsed,/Turn on microphone/);assert.match(collapsed,/dock-handle/);assert.doesNotMatch(collapsed,/Mute audio output|Send message/);
 const mini=renderToStaticMarkup(React.createElement(DockControls,{...props,form:'chat-message-form'}));
 assert.equal((mini.match(/<button/g)||[]).length,4);assert.match(mini,/Mute audio output/);assert.match(mini,/form="chat-message-form"/);
});
test('compact bubbles include timestamps and backend-supplied annotations without interruption boilerplate',()=>{
 const html=renderToStaticMarkup(React.createElement(Message,{compact:true,message:{role:'assistant',text:'Reply',timestamp:1700000000,interrupted:true,cutoff:5,interjections:[{offset:2,text:'legacy prefix',display_text:'My words',system_label:'Speaking over you',started_at:1700000000}]}}));
 assert.match(html,/system-extension/);assert.match(html,/Speaking over you/);assert.match(html,/My words/);assert.match(html,/bubble-time/);assert.doesNotMatch(html,/legacy prefix|Interrupted —/);
 const plain=renderToStaticMarkup(React.createElement(Message,{compact:true,message:{role:'user',text:'[speaking over you] is literal text'}}));
 assert.doesNotMatch(plain,/system-extension/);
});
