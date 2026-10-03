/** Selection keys avoid stale cached model bytes and do not accept arbitrary asset URLs. */
export function avatarModelURL(api,model,format='auto'){
  return api+'/api/avatar/model?selection='+encodeURIComponent(model||'')+'&format='+encodeURIComponent(format);
}

export function validateAvatarFormat(metaVersion,format='auto'){
  const detected=String(metaVersion)==='0'?'vrm0':String(metaVersion)==='1'?'vrm1':null;
  if(!detected)throw new Error('Unsupported VRM model version');
  if(!['auto','vrm0','vrm1'].includes(format)||format!=='auto'&&format!==detected)throw new Error('VRM format does not match the selected model');
  return detected;
}
