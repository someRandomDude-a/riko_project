/** One animation lane; stale loads never replay a replaced/expired wake action. */
export function createWakeAnimationPlayer({load, createMixer, createClip, dispose, report}) {
  let current = null, mixer = null, playback = null, generation = 0, closed = false;
  function stop() {
    if (mixer) {mixer.stopAllAction(); mixer.uncacheRoot(mixer.getRoot());}
    mixer = playback = null;
  }
  function update(action, vrm, delta) {
    if (closed) return;
    if ((action?.id || null) !== current) {
      stop(); current = action?.id || null;
      const version = ++generation;
      if (action && vrm) {
        Promise.resolve().then(() => load(action.payload.path)).then(asset => {
          try {
            if (closed || version !== generation) return;
            const animation = asset.userData?.vrmAnimations?.[0];
            if (!animation) throw new Error('Asset has no VRM animation');
            const clip = createClip(animation, vrm);
            mixer = createMixer(vrm.scene);
            playback = mixer.clipAction(clip);
            mixer.addEventListener('finished', () => {
              if (version !== generation || closed) return;
              stop(); report(action.id, 'completed');
            });
            playback.clampWhenFinished = false;
            playback.play();
            report(action.id, 'started');
          } finally {dispose(asset);}
        }).catch(error => {
          if (!closed && version === generation) {stop(); report(action.id, 'error', error.message);}
        });
      }
    }
    mixer?.update(delta);
  }
  return {update, get playing() {return !!playback;}, dispose() {closed = true; ++generation; stop();}};
}
