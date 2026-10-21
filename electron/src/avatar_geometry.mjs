export function cameraDistance(size, aspect, fov = 30) {
  const tan = Math.tan(fov * Math.PI / 360);
  return Math.max(size.y / (2 * tan), size.x / (2 * tan * aspect)) * 1.18 + size.z / 2;
}

export function resizeHeldAvatar(rect, delta, pointer, viewport) {
  const factor = Math.max(100 / rect.width, 100 / rect.height,
    Math.min(Math.exp(-delta * .0015), viewport.width / rect.width, viewport.height / rect.height, 4096 / rect.width, 4096 / rect.height));
  const width = Math.round(rect.width * factor), height = Math.round(rect.height * factor);
  return {...rect, width, height,
    x: Math.round(Math.max(0, Math.min(viewport.width - width, pointer.x - (pointer.x - rect.x) * factor))),
    y: Math.round(Math.max(0, Math.min(viewport.height - height, pointer.y - (pointer.y - rect.y) * factor)))};
}

/** The scruff slide (an offset in canvas NDC, y up) in whole screen px (y down) for a canvas of this size. */
export function slidePixels(slide, size) {
  return slide ? {x: Math.round(slide.x * size.width / 2), y: Math.round(-slide.y * size.height / 2)} : {x: 0, y: 0};
}

/** Where a body drag puts the canvas: its base (left, top) moved with the cursor since the press, plus the slide in px. */
export function dragPosition(drag, point, slid) {
  return {x: drag.left + point.x - drag.x + slid.x, y: drag.top + point.y - drag.y + slid.y};
}

/** A wheel resize mid-drag: the canvas resized about the cursor, and the drag rebased there with the slide (kept in NDC) re-measured at
 * the new size, so dragPosition gives the resized geometry until the cursor moves and the slide still converges under it. */
export function resizeDrag(drag, delta, pointer, viewport, slide) {
  const geometry = resizeHeldAvatar(drag.geometry, delta, pointer, viewport), slid = slidePixels(slide, geometry);
  return {...drag, x: pointer.x, y: pointer.y, point: {...pointer}, left: geometry.x - slid.x, top: geometry.y - slid.y, geometry, slid};
}
