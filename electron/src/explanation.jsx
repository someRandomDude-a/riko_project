import React,{createContext,useContext} from 'react';
export const ExplanationContext=createContext(false);
export default function Explanation({simple,children}){
  const advanced=useContext(ExplanationContext);
  return <div className="explanation"><p className="caption">{simple}</p>{children&&<details open={advanced||undefined} className="technical-explanation"><summary>Technical details</summary>{children}</details>}</div>;
}
