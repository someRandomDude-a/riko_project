import assert from 'node:assert/strict';
import {test} from 'node:test';
import {normalizeLatex} from './format_text.mjs';
test('supports LaTeX inline and display delimiters', () => {
  assert.equal(normalizeLatex('\\(x^2\\)'), '$x^2$');
  assert.equal(normalizeLatex('\\[x^2\\]'), '\n$$\nx^2\n$$\n');
  assert.equal(normalizeLatex('$x^2$'), '$x^2$');
});
test('leaves code, including unfinished streamed fences, unchanged', () => {
  assert.equal(normalizeLatex('`\\(literal\\)`'), '`\\(literal\\)`');
  assert.equal(normalizeLatex('```python\n\\[literal'), '```python\n\\[literal');
});
test('images: local paths through /api/media, http(s) only as links, everything else as alt text', async () => {
  const {imageSource} = await import('./format_text.mjs');
  assert.deepEqual(imageSource('character_files/a.png'), {kind: 'media', path: 'character_files/a.png'});
  assert.deepEqual(imageSource('/data/character_files/a.png'), {kind: 'media', path: '/data/character_files/a.png'});
  assert.deepEqual(imageSource('C:\\data\\character_files\\a.png'), {kind: 'media', path: 'C:\\data\\character_files\\a.png'});
  assert.deepEqual(imageSource('https://example.com/x.png?q=1'), {kind: 'link', href: 'https://example.com/x.png?q=1', host: 'example.com'});
  assert.deepEqual(imageSource('HTTPS://Example.com/x.png'), {kind: 'link', href: 'https://example.com/x.png', host: 'example.com'});
  assert.deepEqual(imageSource('character_files/my%20image.png'), {kind: 'media', path: 'character_files/my image.png'});  // as react-markdown passes it
  assert.equal(imageSource('http://127.0.0.1:9/x.png').kind, 'link');  // even loopback: an <img> would carry no token anyway
  for (const value of ['//attacker/x.png', '\\\\attacker\\share\\x.png', '/\\attacker\\x.png', 'data:image/png;base64,AAAA',
    'file:///etc/passwd', 'javascript:alert(1)', '%5C%5Cattacker%5Cshare%5Cx.png', '%2F%2Fattacker/x.png', 'a%0Ab.png', 'blob:x', 'ftp://h/x.png', 'a\nb.png', '', undefined, null, 42, 'https://'])
    assert.deepEqual(imageSource(value), {kind: 'none'}, String(value));
});
