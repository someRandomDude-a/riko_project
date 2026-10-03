import React,{useEffect,useState} from 'react';
import {request} from './api.mjs';
import useResource from './use_resource.jsx';
function useApprovals(){
  return useResource('approvals');
}
export function ApprovalBubble({overlay=false}){
  const [data,,error,setError]=useApprovals(),[busy,setBusy]=useState(false);
  const item=data?.pending?.[0];
  useEffect(()=>{if(!overlay)return;window.approvalBridge?.interactive(!!item);return()=>window.approvalBridge?.interactive(false);},[overlay,!!item]);
  async function decide(approved){setBusy(true);try{await request('/api/tools/approvals/'+item.id,{method:'POST',body:{approved}});setError('');}catch(e){setError(e.message);}finally{setBusy(false);}}
  if(!item)return null;
  return <aside className={'approval-bubble '+(overlay?'overlay-approval':'')} role="dialog" aria-label="Approve tool use"><span className="eyebrow">PERMISSION REQUEST{data.pending.length>1?` · ${data.pending.length} waiting`:''}</span><h3>Allow {item.name}?</h3><p>This call will only run after you approve it.</p><details><summary>Review arguments</summary><pre>{JSON.stringify(item.arguments,null,2)}</pre></details>{error&&<p role="alert">{error}</p>}<div className="approval-actions"><button disabled={busy} onClick={()=>decide(false)}>Deny</button><button className="primary" disabled={busy} onClick={()=>decide(true)}>Approve once</button></div></aside>;
}
export function ToolApprovalSettings(){
  const [data,setData,error,setError]=useApprovals(),[busy,setBusy]=useState(false);
  async function change(name,value){setBusy(true);try{const updated=await request('/api/tools/approvals',{method:'PUT',body:{policy:{[name]:value}}});setData(old=>({...old,...updated}));setError('');}catch(e){setError(e.message);}finally{setBusy(false);}}
  return <section className="settings-subsection"><h3>Tool permissions</h3><p className="caption">Require approval before each call. Applies immediately and persists across restarts. New tools follow the global approval default.</p>{error&&<p role="alert">{error}</p>}<div className="settings-grid">{data?.tools.map(tool=><label className="setting-field toggle-field" key={tool.name}><span>{tool.name}<small>{tool.description}</small></span><input type="checkbox" role="switch" disabled={busy} checked={data.policy[tool.name]??data.default_required} onChange={e=>change(tool.name,e.target.checked)}/></label>)}</div></section>;
}
