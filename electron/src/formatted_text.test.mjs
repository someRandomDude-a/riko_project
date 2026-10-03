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
