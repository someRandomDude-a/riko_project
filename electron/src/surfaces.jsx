import React, {useEffect, useRef} from 'react';

import {mediaURL, reportSurface} from './api.mjs';

export function VideoEffect({effect}) {
  const canvas = useRef(null);
  useEffect(() => {
    if (!effect) return;
    let disposed = false, frame, deadline, last = 0, acknowledged = false;
    const video = document.createElement('video');
    video.crossOrigin = 'anonymous'; video.muted = true; video.playsInline = true;
    const context = canvas.current.getContext('2d', {willReadFrequently: true});
    function finish(status, error = '') {
      if (disposed) return;
      disposed = true; cancelAnimationFrame(frame); clearTimeout(deadline); video.pause();
      context.clearRect(0, 0, canvas.current.width, canvas.current.height);
      reportSurface('effect', effect.id, status, error);
    }
    video.onerror = () => finish('error', 'Video unavailable or codec unsupported');
    video.onended = () => finish('completed');
    function draw(now) {
      if (disposed) return;
      try {
        if (video.readyState >= 2 && now - last >= 33) {
          last = now;
          const width = Math.min(1280, video.videoWidth);
          const height = Math.round(video.videoHeight * width / video.videoWidth);
          if (canvas.current.width !== width || canvas.current.height !== height) {canvas.current.width = width; canvas.current.height = height;}
          context.drawImage(video, 0, 0, width, height);
          const image = context.getImageData(0, 0, width, height), pixels = image.data;
          for (let i = 0; i < pixels.length; i += 4) {
            const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
            const dominance = g - Math.max(r, b);
            const alpha = g > 70 ? Math.max(0, Math.min(1, (55 - dominance) / 30)) : 1;
            pixels[i] = Math.min(255, r * effect.brightness);
            pixels[i + 1] = Math.min(255, (alpha < 1 ? Math.min(g, Math.max(r, b)) : g) * effect.brightness);
            pixels[i + 2] = Math.min(255, b * effect.brightness);
            pixels[i + 3] = Math.round(pixels[i + 3] * alpha * effect.opacity);
          }
          context.putImageData(image, 0, 0);
          if (!acknowledged) {
            acknowledged = true; reportSurface('effect', effect.id, 'playing');
            clearTimeout(deadline); deadline = setTimeout(() => finish('completed'), effect.duration * 1000);
          }
        }
        frame = requestAnimationFrame(draw);
      } catch (error) {finish('error', error.message);}
    }
    video.src = mediaURL(effect.asset);
    deadline = setTimeout(() => finish('error', 'Video loading timed out'), 15000);
    video.play().then(() => {frame = requestAnimationFrame(draw);}).catch(error => finish('error', error.message));
    return () => {disposed = true; cancelAnimationFrame(frame); clearTimeout(deadline); video.pause(); video.removeAttribute('src'); video.load(); context.clearRect(0, 0, canvas.current?.width || 0, canvas.current?.height || 0);};
  }, [effect?.id]);
  return <canvas ref={canvas} className="effect-canvas"/>;
}
