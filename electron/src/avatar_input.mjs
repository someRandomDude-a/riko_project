export class AvatarGesture{
 constructor(){this.press=null;}
 down(point,target){this.press={origin:{...point},point:{...point},target,dragging:false};}
 move(point,threshold=6){if(!this.press)return false;this.press.point={...point};if(!this.press.dragging&&Math.hypot(point.x-this.press.origin.x,point.y-this.press.origin.y)>=threshold){this.press.dragging=true;return true;}return false;}
 up(){const result=this.press;this.press=null;return result;}
}
export class AvatarHover{
 constructor(){this.target=null;this.since=0;this.delayed=false;}
 update(target,now,delay){const events=[];if(target?.bone!==this.target?.bone){if(this.target)events.push({kind:'leave',target:this.target});this.target=target;this.since=now;this.delayed=false;if(target)events.push({kind:'hover',target});}if(target&&!this.delayed&&now-this.since>=delay){this.delayed=true;events.push({kind:'hoverDelayed',target:this.target});}return events;}
}
