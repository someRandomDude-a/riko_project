export function normalizeLatex(text) {
  return text.split(/(```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`[^`]*`)/g).map(part => {
    if (part.startsWith('`') || part.startsWith('~~~')) return part;
    return part.replace(/\\\[([\s\S]*?)\\\]/g, (_, math) => `\n$$\n${math}\n$$\n`).replace(/\\\(([\s\S]*?)\\\)/g, (_, math) => `$${math}$`);
  }).join('');
}
