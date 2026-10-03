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
