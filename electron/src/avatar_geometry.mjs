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
