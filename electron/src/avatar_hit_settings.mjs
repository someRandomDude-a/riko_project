export const hitDefaults={avatarHitOutlines:false,avatarHitBones:[],avatarHitRules:{}};
export const hitAnimations=['default','held','clicked','settling','idle','listening','thinking','speaking','playful','recovering'];
export const hitEvents=['click','hold','release','hover','leave','hoverDelayed','ragdoll','recover'];
export const hitExpressions=['default','happy','sad','angry','surprised','neutral'];
const hierarchyBone=bone=>typeof bone==='string'&&/^root(?:\/\d+){1,128}$/.test(bone);
export function normalizeHitSettings(raw,bones){
 const rules={};
 for(const [bone,events] of Object.entries(raw.avatarHitRules||{})){
    if((!bones.includes(bone)&&!hierarchyBone(bone))||!events||typeof events!=='object')continue;
  const normalized={};
  for(const event of hitEvents){
   const rule=events[event];if(!rule||typeof rule!=='object')continue;
   normalized[event]={animation:hitAnimations.includes(rule.animation)?rule.animation:'default',expression:hitExpressions.includes(rule.expression)?rule.expression:'default',intensity:typeof rule.intensity==='number'&&Number.isFinite(rule.intensity)?Math.max(0,Math.min(1,rule.intensity)):.7};
   if(typeof rule.assetId==='string'&&/^[\w-]{1,128}$/.test(rule.assetId))normalized[event].assetId=rule.assetId;
   if(['none','warm','cool','monochrome'].includes(rule.effect))normalized[event].effect=rule.effect;
  }
  rules[bone]=normalized;
 }
  return {avatarHitOutlines:raw.avatarHitOutlines===true,avatarHitBones:Array.isArray(raw.avatarHitBones)?[...new Set(raw.avatarHitBones.filter(bone=>bones.includes(bone)||hierarchyBone(bone)))]:[],avatarHitRules:rules};
}
