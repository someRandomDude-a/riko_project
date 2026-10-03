import {normalizeProfile,normalizeBone} from './avatar_studio_settings.mjs';
import {normalizeGraphics} from './avatar_graphics.mjs';
import {simpleLabel,simpleHelp} from './plain_language.mjs';
import {parseSetting} from './settings_model.mjs';

const studioNumbers={
 mouseSphere:[['dragSpring','Spring drag strength',0,100],['radius','Sphere radius (m)',.01,1],['depth','Sphere front depth (m)',-.5,.5],['strength','Sphere push strength',0,50],['frequency','Mouse follow frequency (Hz)',.5,20],['damping','Mouse follow damping',.1,2],['bounce','Click Z bounce (m)',0,.5],['bounceSeconds','Click bounce duration (s)',.1,2],['maxAngle','Sphere joint angular limit (degrees)',0,90]],
 gaze:[['strength','Convergence strength',0,3],['smoothing','Gaze smoothing response',1,30],['minDistance','Minimum eye target distance (m)',.05,1],['maxYaw','Maximum gaze yaw (degrees)',0,90],['maxPitch','Maximum gaze pitch (degrees)',0,60],['maxConvergence','Maximum convergence per eye (degrees)',0,30]],
 pickup:[['gravity','Gravity',0,30],['damping','Pickup damping',0,10],['bodySpring','Body follow spring',1,80],['inertia','Drag inertia',0,3],['maxSwing','Maximum body swing (degrees)',0,180],['jointLimit','Joint angular limit (degrees)',0,120],['distanceThreshold','Ragdoll distance threshold (px)',10,3000],['speedThreshold','Ragdoll speed threshold (px/s)',50,10000],['recoverySeconds','Recovery duration (seconds)',.2,10]],
 input:[['dragThreshold','Drag activation distance (px)',2,50],['hoverDelay','Delayed hover (ms)',0,5000]]
};
const studioToggles=[['debug','Show skeleton debug view'],['tPose','Preview calibrated T-pose (pauses animation playback)'],['mouseSphere.enabled','Enable mouse force sphere'],['mouseSphere.debug','Show mouse sphere'],['gaze.enabled','Enable mouse eye tracking'],['gaze.convergence','Enable distance-based pupil convergence'],['pickup.enabled','Enable articulated pickup'],['pickup.ragdollEnabled','Enable threshold ragdoll'],['effect.glslEnabled','Enable imported GLSL (trusted GPU code only)']];
const boneNumbers=[['amount','Motion blend',0,1],['frequency','Spring frequency (Hz)',.2,20],['damping','Damping ratio',0,2],['maxAngle','Maximum angular lag (degrees)',0,90]];
const springNumbers=[['stiffness','VRM stiffness',0,10],['dragForce','VRM drag',0,1],['gravityPower','VRM gravity strength',0,5],['hitRadius','VRM collision radius (m)',0,.5]];
const get=(object,path)=>path.split('.').reduce((value,key)=>value?.[key],object);
const set=(object,path,value)=>{const [key,...rest]=path.split('.');return {...object,[key]:rest.length?set(object?.[key]||{},rest.join('.'),value):value};};

export function settingsIndex(fields,groups,catalog){
 const items=fields.map(field=>({...field,id:'runtime:'+field.path,label:simpleLabel(field),technicalLabel:field.label,simpleHelp:simpleHelp(field),source:'runtime',target:{group:field.group,path:field.path}}));
 const local=(path,label,kind,group,options={})=>items.push({id:'local:'+path,path,label,kind,group,source:'local',section:group==='appearance'?'Avatar physics, calibration & effects':group==='graphics'?'Avatar graphics':'Chat layout',target:{group,label,sectionId:group==='appearance'?'appearance-avatar-studio':group==='graphics'?'settings-graphics':'settings-interface'},...options});
 for(const [path,label] of [['activity','Open activity panel'],['tools','Show tool calls'],['reasoning','Show provider reasoning'],['system','Show system events']])local(path,label,'boolean','interface');
 local('showTaskActions','Show active task shortcuts','boolean','interface',{target:{group:'interface',label:'Show active task shortcuts',sectionId:'settings-shortcuts'}});
 local('density','Spacing','string','interface',{options:['comfortable','compact']});
 local('avatarHitOutlines','Show hit-volume outlines','boolean','appearance',{section:'Interaction connections',target:{group:'appearance',label:'Show hit-volume outlines',sectionId:'appearance-avatar-hits'}});
 local('avatarGraphics.antialias','Canvas anti-aliasing','boolean','graphics');
 for(const [key,label,options] of [['samples','Effect-pass MSAA samples',[0,2,4,8]],['anisotropy','Anisotropic texture filtering',[1,2,4,8,16]],['pixelRatio','Maximum render pixel ratio',[1,1.5,2,3]]])local('avatarGraphics.'+key,label,'number','graphics',{options});
 if(catalog){
  const profile=(path,label,kind,options={})=>local('profile.'+path,label,kind,'appearance',{profilePath:path,...options});
  profile('springPickRadius','Spring selection radius (m)','number',{min:.005,max:.2,help:'Selection padding only; does not change physics colliders.'});
  for(const [path,label] of studioToggles)profile(path,label,'boolean');
  for(const [section,numbers] of Object.entries(studioNumbers))for(const [key,label,min,max] of numbers)profile(section+'.'+key,label,'number',{min,max,section:section==='gaze'?'Eye tracking & pupil convergence':section==='pickup'?'Pickup, ragdoll & recovery':section==='input'?'Mouse gestures':'Spring dragging & mouse sphere'});
  for(const [key,label,min,max] of [['ambient','Ambient intensity',0,5]])profile('lighting.'+key,label,'number',{min,max,section:'Model-space lighting'});
  for(const [key,label,min,max] of [['exposure','Exposure multiplier',0,3],['saturation','Color saturation',0,2]])profile('effect.'+key,label,'number',{min,max,section:'Shader effects'});
  profile('effect.preset','Color effect','string',{options:['none','warm','cool','monochrome'],section:'Shader effects'});
  for(const bone of catalog.bones){
   const add=(key,label,kind,options={})=>profile('bones.'+bone.id+'.'+key,(bone.human||bone.name)+' · '+label,kind,{boneId:bone.id,boneKey:key,boneDefaults:bone.settings,section:'Bone · '+(bone.human||bone.name)+' · '+bone.id,target:{group:'appearance',label,sectionId:'appearance-avatar-studio',boneId:bone.id},...options});
   add('enabled','Enable secondary motion','boolean');
   for(const [key,label,min,max] of boneNumbers)add(key,label,'number',{min,max});
   if(bone.spring){add('springOverride','Override imported spring parameters','boolean');for(const [key,label,min,max] of springNumbers)add(key,label,'number',{min,max,help:'Enable the imported spring parameter override to apply this value.'});}
  }
 }
 for(const [group,label] of groups)items.push({id:'category:'+group,label,group,kind:'section',source:'section',section:'Category',target:{group}});
 for(const [group,label,sectionId,keywords] of [['appearance','Model & animation library','desktop-model-library','VRM VRMA import model animation clips'],['appearance','Interaction connections','appearance-avatar-hits','click hold release hover delayed expressions outline filters'],['appearance','Physics, calibration & effects','appearance-avatar-studio','spring bones T-pose lighting GLSL ragdoll eye gaze IPD'],['interface','Conversation shortcuts','settings-shortcuts','quick actions message shortcuts'],['voice','Live calibration & detector testing','settings-voice-live','microphone wake word VAD'],['initiative','Live initiative preferences & event rules','settings-initiative-live','proactive conversation'],['appearance','Live monitor & surface placement','settings-display-live','display monitor position'],['tools','Tool approvals & permissions','settings-tool-approvals','approval safety tasks']])items.push({id:'section:'+sectionId,label,group,kind:'section',source:'section',help:keywords,target:{group,sectionId}});
  return items;
}
export function searchSettings(items,query,groups=[]){
 const tokens=query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean),labels=Object.fromEntries(groups);
 if(!tokens.length)return items.filter(item=>item.source==='section');
 return items.map((item,index)=>{const label=item.label.toLocaleLowerCase(),text=[item.label,item.technicalLabel,item.path,item.help,item.simpleHelp,item.section,labels[item.group]].join(' ').toLocaleLowerCase();return {item,index,score:tokens.every(token=>text.includes(token))?tokens.reduce((score,token)=>score+(label.startsWith(token)?3:label.includes(token)?2:1),0):0};}).filter(match=>match.score).sort((a,b)=>b.score-a.score||a.index-b.index).map(match=>match.item);
}
export function localSettingValue(item,preferences,catalog){
 if(item.profilePath){const profile=normalizeProfile(preferences.avatarStudioProfiles?.[catalog?.key]);return item.boneId?(profile.bones[item.boneId]||normalizeBone(item.boneDefaults))[item.boneKey]:get(profile,item.profilePath);}
 if(item.path.startsWith('avatarGraphics.'))return get(normalizeGraphics(preferences),item.path);
 return get(preferences,item.path);
}
export function localSettingPatch(item,input,preferences,catalog){
 const parsed=parseSetting(item,input);if(parsed.error)return parsed;
 if(item.options&&!item.options.includes(parsed.value))return {error:'Choose a supported value'};
 if(item.profilePath){
  if(!catalog)return {error:'Load an avatar before editing model-specific settings'};
  const profile=normalizeProfile(preferences.avatarStudioProfiles?.[catalog.key]);
  const next=item.boneId?{...profile,bones:{...profile.bones,[item.boneId]:normalizeBone({...profile.bones[item.boneId]||normalizeBone(item.boneDefaults),[item.boneKey]:parsed.value})}}:set(profile,item.profilePath,parsed.value);
  return {changes:{avatarStudioProfiles:{...preferences.avatarStudioProfiles,[catalog.key]:next}}};
 }
 return {changes:item.path.startsWith('avatarGraphics.')?{avatarGraphics:set(normalizeGraphics(preferences).avatarGraphics,item.path.slice(15),parsed.value)}:set({},item.path,parsed.value)};
}

export function focusSetting(root,target){
 const section=target.sectionId?root.querySelector('#'+target.sectionId):root.querySelector('#settings-panel');
 if(section?.tagName==='DETAILS')section.open=true;
 const field=target.path?root.querySelector('#setting-'+target.path.replaceAll('.','-')):target.label?Array.from((section||root).querySelectorAll('label')).find(label=>label.textContent.trim().startsWith(target.label))?.querySelector('input,select,textarea'):null;
 const node=field||section||root;node.scrollIntoView({block:'center',behavior:'auto'});if(!field)node.setAttribute('tabindex','-1');node.focus({preventScroll:true});
 return node;
}
