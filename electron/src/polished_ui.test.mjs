import test,{before,after} from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {createServer} from 'vite';
import {normalizePreferences} from './ui_preferences.mjs';
import {settingsIndex} from './settings_search.mjs';
import {readFile} from 'node:fs/promises';
let server,Explanation,Context,Appearance,Tabs,DockControls,Message,StreamChat,AvatarStudio,DockVolume,SettingsPage,SettingsSearchResults;
before(async()=>{
 server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom'});
 ({default:Explanation,ExplanationContext:Context}=await server.ssrLoadModule('/src/explanation.jsx'));
 ({default:Appearance}=await server.ssrLoadModule('/src/appearance_page.jsx'));
 ({SettingsTabs:Tabs}=await server.ssrLoadModule('/src/settings_navigation.jsx'));
 ({default:DockControls}=await server.ssrLoadModule('/src/dock_controls.jsx'));
  ({Message}=await server.ssrLoadModule('/src/stream_chat.jsx'));
  ({default:StreamChat}=await server.ssrLoadModule('/src/stream_chat.jsx'));
  ({default:AvatarStudio}=await server.ssrLoadModule('/src/avatar_studio.jsx'));
  ({DockVolume}=await server.ssrLoadModule('/src/dock_audio.jsx'));
  ({default:SettingsPage}=await server.ssrLoadModule('/src/settings_page.jsx'));
  ({SettingsSearchResults}=await server.ssrLoadModule('/src/settings_search.jsx'));
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
test('collapsed dock has only the microphone and bottom bar; mini chat adds mute and send',()=>{
 const props={preferences:normalizePreferences({})};
 const collapsed=renderToStaticMarkup(React.createElement(DockControls,{...props,collapsed:true}));
 assert.equal((collapsed.match(/<button/g)||[]).length,2);
 assert.match(collapsed,/Turn on microphone/);assert.match(collapsed,/dock-handle/);assert.doesNotMatch(collapsed,/Mute audio output|Send message/);
 const mini=renderToStaticMarkup(React.createElement(DockControls,{...props,form:'chat-message-form'}));
 assert.equal((mini.match(/<button/g)||[]).length,4);assert.match(mini,/Mute audio output/);assert.match(mini,/form="chat-message-form"/);
});
test('full and mini chats use the same dock with a single send button',()=>{
 const previous=globalThis.window;globalThis.window={innerWidth:1280,innerHeight:720};
 try{
 const render=compactMode=>renderToStaticMarkup(React.createElement(StreamChat,{compactMode,preferences:normalizePreferences({})}));
 for(const compact of [false,true]){const html=render(compact);assert.match(html,/Mute audio output/);assert.match(html,/class="dock-mic /);assert.equal((html.match(/aria-label="Send message"/g)||[]).length,1);assert.match(html,/form="chat-message-form"/);assert.doesNotMatch(html,/mic-only|dock-collapsed|class="send-button"/);}
 assert.match(render(false),/aria-label="Start Discord client"/);
 }finally{if(previous===undefined)delete globalThis.window;else globalThis.window=previous;}
});
test('compact bubbles include timestamps and backend-supplied annotations without interruption boilerplate',()=>{
 const html=renderToStaticMarkup(React.createElement(Message,{compact:true,message:{role:'assistant',text:'Reply',timestamp:1700000000,interrupted:true,cutoff:5,interjections:[{offset:2,text:'legacy prefix',display_text:'My words',system_label:'Speaking over you',started_at:1700000000}]}}));
 assert.match(html,/system-extension/);assert.match(html,/Speaking over you/);assert.match(html,/My words/);assert.match(html,/bubble-time/);assert.doesNotMatch(html,/legacy prefix|Interrupted —/);
 const plain=renderToStaticMarkup(React.createElement(Message,{compact:true,message:{role:'user',text:'[speaking over you] is literal text'}}));
 assert.doesNotMatch(plain,/system-extension/);
});
test('Desktop model exposes distinct numeric and slider controls for spring drag strength',()=>{
 const previous=globalThis.localStorage;
 globalThis.localStorage={getItem:()=>JSON.stringify({key:'test.vrm#test',source:'test.vrm',bones:[{id:'root/0',name:'hair',spring:true}]})};
 try{
  const preferences=normalizePreferences({avatarStudioProfiles:{'test.vrm#test':{mouseSphere:{dragSpring:75}}}});
  const html=renderToStaticMarkup(React.createElement(AvatarStudio,{preferences,update:()=>{}}));
  assert.match(html,/aria-label="Spring drag strength" type="number"[^>]*value="75"/);
  assert.match(html,/aria-label="Spring drag strength slider" type="range"[^>]*value="75"/);
  assert.match(html,/Sphere push strength/);assert.match(html,/Dragging works even when mouse sphere pushing is disabled/);
 }finally{if(previous===undefined)delete globalThis.localStorage;else globalThis.localStorage=previous;}
});
test('volume popover is a horizontal fading row with no mute hint text',async()=>{
 const html=renderToStaticMarkup(React.createElement(DockVolume,{id:'volume-test',gain:.65,onChange:()=>{}}));
 assert.match(html,/aria-orientation="horizontal"/);assert.match(html,/>65%<\/output>/);assert.doesNotMatch(html,/click audio again|mute|unmute|<label/i);
 const css=await readFile(new URL('./ui/dock.css',import.meta.url),'utf8');
 assert.match(css,/\.dock-volume-panel\{[^}]*display:flex;align-items:center/);assert.match(css,/animation:dock-volume-in/);assert.match(css,/@keyframes dock-volume-in\{from\{opacity:0\}to\{opacity:1\}\}/);
});
test('Settings page exposes a keyboard-accessible search menu without changing runtime drafts',()=>{
 const html=renderToStaticMarkup(React.createElement(SettingsPage,{preferences:normalizePreferences({}),updatePreferences:()=>{}}));
 assert.match(html,/aria-label="Open settings search menu" aria-haspopup="dialog"/);assert.match(html,/Ctrl K/);assert.match(html,/Advanced controls/);
});
test('search results offer navigation and inline editors while honoring read-only fields',()=>{
 const catalog={key:'test.vrm#search',bones:[]},preferences=normalizePreferences({});
 const fields=[{path:'runtime.n_ctx',label:'Context',group:'models',kind:'number',advanced:true},{path:'runtime.version',label:'Version',group:'models',kind:'string',readonly:true}];
 const indexed=settingsIndex(fields,[['models','Models'],['appearance','Desktop model']],catalog);
 const matches=[...indexed.filter(i=>i.source==='runtime'),indexed.find(i=>i.path==='profile.mouseSphere.dragSpring')];
 let rendered=0;const html=renderToStaticMarkup(React.createElement(SettingsSearchResults,{matches,active:0,labels:{models:'Models',appearance:'Desktop model'},preferences,catalog,updatePreferences:()=>{},onNavigate:()=>{},renderRuntime:(field,prefix)=>{rendered++;return React.createElement('input',{id:prefix+field.path,value:4096,readOnly:true});}}));
 assert.equal(rendered,1);assert.equal((html.match(/class="settings-search-jump"/g)||[]).length,3);
 assert.match(html,/Edit here · runtime draft/);assert.match(html,/Edit here · applies immediately/);assert.match(html,/Read-only · open to inspect/);assert.match(html,/Advanced/);assert.match(html,/value="25"/);
});
