import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import {normalizeLatex} from './format_text.mjs';
import {mediaURL} from './api.mjs';

export default function FormattedText({text = ''}) {
  const normalized = normalizeLatex(text);
  return <div className="formatted-text"><ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[[rehypeKatex, {strict: false, throwOnError: false}]]} components={{a: ({children, href}) => <a href={href} target="_blank" rel="noreferrer">{children}</a>, img: ({src,alt}) => <img alt={alt} src={src && !/^https?:\/\//i.test(src) ? mediaURL(src) : src}/>}}>{normalized}</ReactMarkdown></div>;
}
