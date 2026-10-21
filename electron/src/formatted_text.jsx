import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import {imageSource, normalizeLatex} from './format_text.mjs';
import {mediaURL} from './api.mjs';

// A remote image is a click-to-open link (opened in the browser by main's window-open handler), never an <img> that loads it;
// a local one comes through /api/media (imageSource in format_text.mjs). index.html's CSP refuses other image origins too.
function MarkdownImage({src, alt}) {
  const source = imageSource(src);
  if (source.kind === 'media') return <img alt={alt} src={mediaURL(source.path)}/>;
  if (source.kind === 'link') return <a className="image-link" href={source.href} target="_blank" rel="noreferrer" title={source.href}>{alt || 'Image'} ({source.host})</a>;
  return alt ? <span className="image-alt">{alt}</span> : null;
}

function MarkdownLink({children, href}) {return <a href={href} target="_blank" rel="noreferrer">{children}</a>;}
// One set of plugins and components for every render: new arrays or an inline component would make react-markdown re-create them
// and React remount every link on each render.
const REMARK = [remarkGfm, remarkMath], REHYPE = [[rehypeKatex, {strict: false, throwOnError: false}]], COMPONENTS = {a: MarkdownLink, img: MarkdownImage};

// Memoized: a message re-parses its Markdown only when its own text changes, never because a sibling streamed a token.
function FormattedText({text = ''}) {
  const normalized = normalizeLatex(text);
  return <div className="formatted-text"><ReactMarkdown remarkPlugins={REMARK} rehypePlugins={REHYPE} components={COMPONENTS}>{normalized}</ReactMarkdown></div>;
}
export default React.memo(FormattedText);
