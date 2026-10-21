export function normalizeLatex(text) {
  return text.split(/(```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`[^`]*`)/g).map(part => {
    if (part.startsWith('`') || part.startsWith('~~~')) return part;
    return part.replace(/\\\[([\s\S]*?)\\\]/g, (_, math) => `\n$$\n${math}\n$$\n`).replace(/\\\(([\s\S]*?)\\\)/g, (_, math) => `$${math}$`);
  }).join('');
}

// What a Markdown image in model, task, whiteboard or error text may load. A local path goes through the backend's scoped
// /api/media; an http(s) URL becomes a link the user clicks, because an <img> would fetch it unasked (text an attacker got
// into a reply could send data out in the URL); anything else (another scheme, a network path such as //host or \\host,
// control characters) renders as its alt text only. react-markdown percent-encodes a source (a backslash is %5C, a space
// %20), so a local path is decoded before it is checked and sent.
export function imageSource(src) {
  if (typeof src !== 'string' || !src || /[\u0000-\u001f\u007f]/.test(src)) return {kind: 'none'};
  if (/^https?:\/\//i.test(src)) {
    try {const url = new URL(src); return {kind: 'link', href: url.href, host: url.host};} catch {return {kind: 'none'};}
  }
  let path = src; try {path = decodeURIComponent(src);} catch {}
  if (!path || /[\u0000-\u001f\u007f]/.test(path) || /^[\\/]{2}/.test(path) || /^[a-z][a-z0-9+.-]*:/i.test(path) && !/^[a-z]:[\\/]/i.test(path)) return {kind: 'none'};
  return {kind: 'media', path};
}
