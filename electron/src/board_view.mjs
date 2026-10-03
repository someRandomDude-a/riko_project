export function readBoardView(key) {
  try {
    const value = JSON.parse(localStorage.getItem(key));
    if (!value || !Number.isFinite(value.view?.x) || !Number.isFinite(value.view?.y) || !Number.isFinite(value.view?.z) || value.view.z < .2 || value.view.z > 4) return null;
    return value;
  } catch {return null;}
}
export function saveBoardView(key, value) {
  try {localStorage.setItem(key, JSON.stringify(value));} catch { /* Content is persisted independently in Python. */ }
}
