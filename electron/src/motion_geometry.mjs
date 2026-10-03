export function nearbyPointer(rect, point, wasNear=false) {
  const x=(point.x-rect.x-rect.width/2)/(rect.width/2);
  const y=(point.y-rect.y-rect.height/2)/(rect.height/2);
  const distance=Math.hypot(x,y);
  return {x:Math.max(-2,Math.min(2,x)),y:Math.max(-2,Math.min(2,y)),near:distance<(wasNear?1.8:1.4)};
}

/** Screen placement is independent of skeletal root motion. */
export class DesktopWalker {
  constructor(report=()=>{}) {this.report=report;this.active=null;this.seen=new Set();this.velocity=0;}
  stop() {
    if(this.active){this.report(this.active.id,'cancelled');this.active=null;}
    this.velocity=0;
  }
  step(action, geometry, viewport, dt, held=false) {
    if(held){if(action)this.seen.add(action.id);this.stop();return null;}
    if(!action){this.stop();return null;}
    if(this.active?.id!==action.id){
      this.stop();if(this.seen.has(action.id))return null;
      this.seen.add(action.id);if(this.seen.size>64)this.seen.delete(this.seen.values().next().value);
      const target={x:Math.max(0,Math.min(viewport.width-geometry.width,action.payload.target.x)),y:Math.max(0,Math.min(viewport.height-geometry.height,action.payload.target.y))};
      this.active={id:action.id,position:{x:geometry.x,y:geometry.y},target,screen:geometry.screen,width:geometry.width,height:geometry.height,speed:action.payload.speed};
      this.report(action.id,'started');
    }
    const move=this.active;
    if(move.screen!==geometry.screen||move.width!==geometry.width||move.height!==geometry.height){this.stop();return null;}
    const dx=move.target.x-move.position.x,dy=move.target.y-move.position.y,distance=Math.hypot(dx,dy);
    const delta=Math.max(0,Math.min(.1,dt)),acceleration=move.speed*4;
    this.velocity=Math.min(move.speed,Math.sqrt(2*acceleration*distance),this.velocity+acceleration*delta);
    const travel=Math.min(distance,this.velocity*delta);
    if(distance<1||travel>=distance){
      const result={...geometry,...move.target,final:true,velocity:0,facing:Math.sign(dx)||1};
      this.active=null;this.velocity=0;this.report(move.id,'completed');return result;
    }
    move.position.x+=dx/distance*travel;move.position.y+=dy/distance*travel;
    return {...geometry,...move.position,final:false,velocity:this.velocity,facing:Math.sign(dx)||1};
  }
}
