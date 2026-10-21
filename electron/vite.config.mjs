import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';

// index.html's Content-Security-Policy, widened only while Vite serves the page (http://localhost:5173 in development) for
// its HMR socket on ws://localhost:<port>. A build keeps the file's policy, so the packaged file:// page never gets it.
export const devCsp = {name: 'riko-dev-csp', apply: 'serve', transformIndexHtml: html => html.replace(/(connect-src [^;"]*)/, '$1 ws://localhost:*')};

export default defineConfig({base: './', plugins: [react, devCsp]});
