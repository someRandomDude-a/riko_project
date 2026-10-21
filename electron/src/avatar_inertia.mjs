import {Vector3} from 'three';
const SMOOTH=.04,JUMP=20000,MAX_ACCELERATION=60000,AIR=1.5,GAIN=.35;  // seconds, px/s (faster is a teleport), px/s², 1/s
// At full physical force (GAIN 1) a brisk drag of a 1.6 m avatar shown 720 px tall is a 20 g jolt that swings hair past its colliders.
const ease=(value,target,dt)=>value+(target-value)*(1-Math.exp(-dt/SMOOTH));

/** The avatar's on-screen motion as an inertial force on its spring bones. Dragging, throwing or walking her moves her
 * window, not her skeleton, and springs centred on the skeleton (VRoid's are) never feel that. So each frame the window's
 * acceleration, plus the air it moves through, becomes a world-space force: hair, skirt and bust lag when she starts
 * moving, stream behind while she moves and bounce when she stops. */
export class AvatarInertia{
 constructor(){this.reset();}
 reset(){this.position=null;this.velocity={x:0,y:0};this.acceleration={x:0,y:0};}
 /** Window position in screen px (y down), once per frame. */
 track(x,y,dt){
  if(!(dt>0)||!Number.isFinite(x)||!Number.isFinite(y))return;
  const last=this.position;this.position={x,y};if(!last)return;
  const vx=(x-last.x)/dt,vy=(y-last.y)/dt;
  if(Math.hypot(vx,vy)>JUMP){this.reset();this.position={x,y};return;}  // moved to another display or re-laid out: not motion
  const velocity={x:ease(this.velocity.x,vx,dt),y:ease(this.velocity.y,vy,dt)};
  let ax=(velocity.x-this.velocity.x)/dt,ay=(velocity.y-this.velocity.y)/dt;const size=Math.hypot(ax,ay);
  if(size>MAX_ACCELERATION){ax*=MAX_ACCELERATION/size;ay*=MAX_ACCELERATION/size;}
  this.velocity=velocity;this.acceleration={x:ease(this.acceleration.x,ax,dt),y:ease(this.acceleration.y,ay,dt)};
 }
 /** The window moved by (dx, dy) px without moving her (a wheel resize about the cursor): the next track() sees no motion from it. */
 shift(dx,dy){if(this.position&&Number.isFinite(dx)&&Number.isFinite(dy))this.position={x:this.position.x+dx,y:this.position.y+dy};}
 /** The force three-vrm adds to a spring tail per update (gravityDir × gravityPower: displacement per second of a
  * frame, so an acceleration a contributes a·dt), in world units, x right and y up. Bounce 0 is none; at 1 a brisk drag
  * swings VRoid hair and bust about 30 degrees, and they settle within a few seconds. */
 force(worldPerPixel,bounce,dt,out=new Vector3()){
  const scale=worldPerPixel*bounce*GAIN*dt;
  return out.set(-(this.acceleration.x+AIR*this.velocity.x)*scale,(this.acceleration.y+AIR*this.velocity.y)*scale,0);
 }
}

/** Run update with force added to every joint's gravity, then put their own gravity back. */
export function withSpringForce(joints,force,update){
 if(!(force.lengthSq()>1e-12))return update();
 const saved=[],vector=new Vector3();
 for(const joint of joints){
  const settings=joint.settings;saved.push([settings,settings.gravityPower,settings.gravityDir.clone()]);
  vector.copy(settings.gravityDir).multiplyScalar(settings.gravityPower).add(force);
  settings.gravityPower=vector.length();if(settings.gravityPower>0)settings.gravityDir.copy(vector).divideScalar(settings.gravityPower);
 }
 try{return update();}finally{for(const [settings,power,direction] of saved){settings.gravityPower=power;settings.gravityDir.copy(direction);}}
}
