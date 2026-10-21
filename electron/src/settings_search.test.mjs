import test from 'node:test';
import assert from 'node:assert/strict';
import {settingsIndex,searchSettings,localSettingValue,localSettingPatch,focusSetting} from './settings_search.mjs';
import {normalizePreferences} from './ui_preferences.mjs';
import {settingsPatch} from './settings_model.mjs';
import {PICKUP_FIELDS,REACTION_FIELDS,FACE_FIELDS,PICKUP_TOGGLES,REACTION_TOGGLES,GAZE_FIELDS,GAZE_TOGGLES,RELEASE_STYLES} from './avatar_studio_settings.mjs';

const groups=[['models','Models'],['appearance','Desktop model'],['graphics','Graphics'],['interface','Chat layout'],['voice','Microphone']];
const catalog={key:'avatar.vrm#one',bones:[{id:'root/0',name:'Hair',spring:true,settings:{stiffness:2,dragForce:.6}},{id:'root/1',name:'Hair',spring:true,settings:{stiffness:3}}]};
const fields=[{path:'runtime.n_ctx',label:'Context capacity',help:'Maximum token budget',section:'Token budgets',group:'models',kind:'number',integer:true,min:1,max:65536,advanced:true},{path:'runtime.api_key',label:'API key',help:'Provider authentication credential',group:'models',kind:'string',secret:true},{path:'runtime.version',label:'Backend version',group:'models',kind:'string',readonly:true}];
const items=settingsIndex(fields,groups,catalog);
const item=path=>items.find(i=>i.path===path);

test('settings search indexes hidden runtime fields, categories, graphics and model-local controls',()=>{
 assert.equal(searchSettings(items,'  spring DRAG strength ',groups)[0].path,'profile.mouseSphere.dragSpring');
 assert.equal(searchSettings(items,'runtime.n_ctx',groups)[0].advanced,true);
 assert.ok(searchSettings(items,'maximum token budget',groups).some(i=>i.path==='runtime.n_ctx'));
 assert.ok(searchSettings(items,'desktop model convergence',groups).some(i=>i.path==='profile.gaze.convergence'));
 assert.ok(searchSettings(items,'anti-aliasing',groups).some(i=>i.path==='avatarGraphics.antialias'));
 assert.ok(searchSettings(items,'resting smile',groups).some(i=>i.path==='profile.face.restingSmile'&&i.section==='Face & reactions'));
 assert.ok(searchSettings(items,'neutral smile',groups).some(i=>i.path==='profile.face.restingSmile'),'the label says it also shows for a neutral or calm emotion');
 assert.ok(searchSettings(items,'reaction expression strength',groups).some(i=>i.path==='profile.face.intensity'));
 assert.ok(searchSettings(items,'  ',groups).every(i=>i.kind==='section'));
 assert.equal(searchSettings(items,'not-a-real-setting',groups).length,0);
 assert.equal(new Set(items.map(i=>i.id)).size,items.length);
});
test('search keeps duplicate-name bones distinct and does not index settings values or credentials',()=>{
 const matches=searchSettings(items,'Hair stiffness',groups);assert.equal(matches.length,2);assert.notEqual(matches[0].boneId,matches[1].boneId);
 assert.equal(searchSettings(items,'root/1 stiffness',groups)[0].boneId,'root/1');
 assert.equal(searchSettings(items,'fake-secret-token',groups).length,0);
 assert.equal(item('runtime.api_key').secret,true);assert.equal(item('runtime.version').readonly,true);
});
test('local inline edits preserve other profiles and nested model preferences',()=>{
 const preferences=normalizePreferences({avatarStudioProfiles:{[catalog.key]:{mouseSphere:{depth:.2},gaze:{enabled:false}},'other.vrm#two':{springPickRadius:.05}}});
 const result=localSettingPatch(item('profile.mouseSphere.dragSpring'),'75',preferences,catalog),next={...preferences,...result.changes};
 assert.equal(localSettingValue(item('profile.mouseSphere.dragSpring'),next,catalog),75);
 assert.equal(next.avatarStudioProfiles[catalog.key].mouseSphere.depth,.2);assert.equal(next.avatarStudioProfiles[catalog.key].gaze.enabled,false);
 assert.equal(next.avatarStudioProfiles['other.vrm#two'].springPickRadius,.05);assert.equal(preferences.avatarStudioProfiles[catalog.key].mouseSphere.dragSpring,25);
 assert.deepEqual(localSettingPatch(item('tools'),false,preferences,catalog).changes,{tools:false});
});
test('local inline editing validates numbers and options before emitting a patch',()=>{
 const p=normalizePreferences({});
 for(const value of ['',Infinity,-1,101,'bad'])assert.ok(localSettingPatch(item('profile.mouseSphere.dragSpring'),value,p,catalog).error);
 assert.ok(localSettingPatch(item('profile.mouseSphere.dragSpring'),25,p,null).error);
 assert.ok(localSettingPatch(item('density'),'bad',p,catalog).error);
 assert.ok(localSettingPatch(item('avatarGraphics.samples'),3,p,catalog).error);
 const next=localSettingPatch(item('avatarGraphics.samples'),8,p,catalog).changes;
 assert.equal(next.avatarGraphics.samples,8);assert.equal(next.avatarGraphics.antialias,p.avatarGraphics.antialias);
});
test('per-bone inline editing starts from imported defaults and preserves other bone overrides',()=>{
 const p=normalizePreferences({avatarStudioProfiles:{[catalog.key]:{bones:{'root/1':{springOverride:true,stiffness:4}}}}});
 const rule=item('profile.bones.root/0.dragForce');assert.equal(localSettingValue(rule,p,catalog),.6);
 const next=localSettingPatch(rule,.2,p,catalog).changes.avatarStudioProfiles[catalog.key];
 assert.equal(next.bones['root/0'].dragForce,.2);assert.equal(next.bones['root/0'].stiffness,2);assert.equal(next.bones['root/1'].stiffness,4);
 assert.equal(rule.target.boneId,'root/0');assert.equal(rule.target.label,'VRM drag');
});
test('runtime search edits remain validated draft values rather than local preference writes',()=>{
 const values={'runtime.n_ctx':4096,'runtime.api_key':'','runtime.version':'1'},inputs={...values,'runtime.n_ctx':'8192'};
 assert.deepEqual(settingsPatch(fields,values,inputs).changes,{'runtime.n_ctx':8192});
 assert.ok(settingsPatch(fields,values,{...inputs,'runtime.n_ctx':'0'}).errors['runtime.n_ctx']);
 assert.equal(item('runtime.n_ctx').source,'runtime');assert.deepEqual(item('runtime.n_ctx').target,{group:'models',path:'runtime.n_ctx'});
});
test('navigation opens collapsed sections and focuses a matching control or section',()=>{
 const calls=[],control={scrollIntoView:options=>calls.push(['scroll',options]),focus:options=>calls.push(['focus',options])};
 const section={tagName:'DETAILS',open:false,querySelectorAll:()=>[{textContent:'Spring drag strength',querySelector:()=>control}]};
 const root={querySelector:selector=>selector==='#appearance-avatar-studio'?section:null};
 assert.equal(focusSetting(root,{sectionId:'appearance-avatar-studio',label:'Spring drag strength'}),control);assert.equal(section.open,true);assert.equal(calls.length,2);
 const pathRoot={querySelector:selector=>selector==='#setting-runtime-n_ctx'?control:section};assert.equal(focusSetting(pathRoot,{path:'runtime.n_ctx'}),control);
});
test('without an avatar catalog the index still offers model section navigation but no stale profile edits',()=>{
 const empty=settingsIndex(fields,groups,null);assert.ok(empty.some(i=>i.id==='section:appearance-avatar-studio'));assert.ok(!empty.some(i=>i.profilePath));
});
test('every shared pickup, reaction, face and gaze field is indexed under its studio section; grab and release styles list their options',()=>{
 for(const [section,label,fields] of [['pickup','Pickup, throwing & landing',PICKUP_FIELDS],['reactions','Held reactions',REACTION_FIELDS],['face','Face & reactions',FACE_FIELDS],['gaze','Eye tracking & pupil convergence',GAZE_FIELDS]])for(const [key,name,min,max] of fields){
  const entry=item('profile.'+section+'.'+key);assert.ok(entry,`${section}.${key} indexed`);assert.equal(entry.label,name);assert.equal(entry.section,label);assert.deepEqual([entry.min,entry.max],[min,max]);
 }
 for(const [key] of PICKUP_TOGGLES)assert.equal(item('profile.pickup.'+key).section,'Pickup, throwing & landing');for(const [key] of REACTION_TOGGLES)assert.equal(item('profile.reactions.'+key).kind,'boolean');
 for(const [key,label] of GAZE_TOGGLES){const entry=item('profile.gaze.'+key);assert.equal(entry.kind,'boolean');assert.equal(entry.label,label);assert.equal(entry.section,'Eye tracking & pupil convergence');}
 const style=searchSettings(items,'grab style',groups)[0];assert.equal(style.path,'profile.pickup.grabStyle');assert.deepEqual(style.options,['scruff','bone']);
 const release=searchSettings(items,'release',groups)[0];assert.equal(release.path,'profile.pickup.release');assert.deepEqual(release.options,RELEASE_STYLES.map(([key])=>key));assert.match(release.help,/flick/);
 for(const [query,path] of [['throw speed','throwSpeed'],['fall gravity','fallGravity'],['landing squash','landingSquash'],['recovery time','recoverySeconds']])assert.equal(searchSettings(items,query,groups)[0].path,'profile.pickup.'+path,query);
 assert.equal(searchSettings(items,'saccades',groups)[0].path,'profile.gaze.saccades');assert.ok(searchSettings(items,'throw',groups).some(i=>i.id==='section:appearance-avatar-studio'),'the section is found by throw');
 assert.ok(searchSettings(items,'scruff tilt',groups).some(i=>i.path==='profile.pickup.scruffPitch'));assert.ok(searchSettings(items,'kick size',groups).some(i=>i.path==='profile.reactions.kickAmplitude'));
 assert.ok(searchSettings(items,'swing damping',groups).some(i=>i.path==='profile.pickup.swingDamping'));assert.equal(items.some(i=>/maxSwing|bodySpring|jointLimit|pickup\.gravity/.test(i.path||'')),false);
});
test('inline edits of a renamed pickup key round-trip without touching other profiles or reviving old keys',()=>{
 const p=normalizePreferences({avatarStudioProfiles:{[catalog.key]:{pickup:{maxSwing:180,swingLimit:20}},'other.vrm#two':{pickup:{carryGain:2}}}});
 const result=localSettingPatch(item('profile.pickup.swingLimit'),'40',p,catalog),next={...p,...result.changes};
 assert.equal(localSettingValue(item('profile.pickup.swingLimit'),next,catalog),40);assert.equal(next.avatarStudioProfiles['other.vrm#two'].pickup.carryGain,2);
 assert.equal('maxSwing' in next.avatarStudioProfiles[catalog.key].pickup,false);
 assert.ok(localSettingPatch(item('profile.pickup.grabStyle'),'tail',p,catalog).error);assert.equal(localSettingPatch(item('profile.pickup.grabStyle'),'bone',p,catalog).changes.avatarStudioProfiles[catalog.key].pickup.grabStyle,'bone');
 assert.ok(localSettingPatch(item('profile.pickup.release'),'yeet',p,catalog).error);assert.equal(localSettingPatch(item('profile.pickup.release'),'place',p,catalog).changes.avatarStudioProfiles[catalog.key].pickup.release,'place');
 const off=localSettingPatch(item('profile.gaze.saccades'),false,p,catalog);assert.equal(off.changes.avatarStudioProfiles[catalog.key].gaze.saccades,false);assert.equal(off.changes.avatarStudioProfiles['other.vrm#two'].pickup.carryGain,2);
});
test('the avatar frame rate is a graphics setting: searchable, adaptive by default, and only adaptive or full',()=>{
 const p=normalizePreferences({}),rate=item('avatarGraphics.frameRate');
 assert.ok(searchSettings(items,'frame rate',groups).some(i=>i.path==='avatarGraphics.frameRate'));
 assert.ok(searchSettings(items,'avatar fps battery',groups).some(i=>i.path==='avatarGraphics.frameRate'));
 assert.equal(rate.target.sectionId,'settings-graphics');assert.equal(localSettingValue(rate,p,catalog),'adaptive');
 assert.ok(localSettingPatch(rate,'60fps',p,catalog).error);
 const next=localSettingPatch(rate,'full',p,catalog).changes;assert.equal(next.avatarGraphics.frameRate,'full');assert.equal(next.avatarGraphics.samples,p.avatarGraphics.samples);
 assert.equal(normalizePreferences({avatarGraphics:{frameRate:'turbo'}}).avatarGraphics.frameRate,'adaptive');
});
