import http from 'node:http';
import { readFile, realpath } from 'node:fs/promises';
import { mkdirSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';
import { Readable } from 'node:stream';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { handle, reserve, settle } from './core.js';

// Local preview only. Production uses worker.js, with persistent atomic limits.
if (process.argv[2]) process.loadEnvFile(process.argv[2]);
const root = fileURLToPath(new URL('../', import.meta.url));
const port = Number(process.env.PORT || 8937);
const localOrigin = `http://127.0.0.1:${port}`;
mkdirSync(path.join(root, '.demo-state'), { recursive: true, mode: 0o700 });
const db = new DatabaseSync(path.join(root, '.demo-state/quota.sqlite'));
db.exec('CREATE TABLE IF NOT EXISTS quota (id INTEGER PRIMARY KEY, entries TEXT NOT NULL)');
db.prepare('INSERT OR IGNORE INTO quota VALUES (1, ?)').run('[]');
function transaction(fn) {
  db.exec('BEGIN IMMEDIATE');
  try {
    const entries = JSON.parse(db.prepare('SELECT entries FROM quota WHERE id=1').get().entries);
    const result = fn(entries);
    db.prepare('UPDATE quota SET entries=? WHERE id=1').run(JSON.stringify(result.entries));
    db.exec('COMMIT');
    return result;
  } catch (error) { db.exec('ROLLBACK'); throw error; }
}
const limiter = {
  reserve: budget => transaction(entries => reserve(entries, 'local-preview', budget, Date.now(), randomUUID())),
  settle: (id, usage) => transaction(entries => ({ entries: settle(entries, id, usage) })),
};
const types = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.json': 'application/json', '.txt': 'text/plain', '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp', '.gz': 'application/gzip', '.woff2': 'font/woff2' };
http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, localOrigin);
    if (url.pathname === '/api/rootcause') {
      const response = await handle(new Request(url, { method: req.method, headers: req.headers, ...(req.method !== 'GET' && req.method !== 'HEAD' ? { body: Readable.toWeb(req), duplex: 'half' } : {}) }), { DEMO_MODE: process.env.DEMO_MODE, GROQ_API_KEY: process.env.GROQ_API_KEY, ALLOWED_ORIGINS: `${localOrigin},http://localhost:${port}` }, limiter);
      res.writeHead(response.status, Object.fromEntries(response.headers));
      res.end(Buffer.from(await response.arrayBuffer()));
      return;
    }
    if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405).end(); return; }
    const name = decodeURIComponent(url.pathname === '/' ? '/index.html' : url.pathname);
    // Explicit public paths: never serve backend source, dotfiles, keys, or quota DB.
    if (!/^\/(index\.html|(?:css|js|assets)\/[\w./-]+|demo\/inputs\/(trace|opguard)\.txt|demo\/sources\/run[12]-stage_1_and_2\.(html|py))$/.test(name) || name.split('/').some(part => part.startsWith('.'))) { res.writeHead(404).end(); return; }
    const filename = await realpath(path.join(root, name));
    if (!filename.startsWith(root)) { res.writeHead(404).end(); return; }
    const data = await readFile(filename);
    res.writeHead(200, { 'Content-Type': types[path.extname(filename)] || 'application/octet-stream', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
    res.end(req.method === 'HEAD' ? undefined : data);
  } catch { res.writeHead(404).end(); }
}).listen(port, '127.0.0.1', () => console.log(`Local: ${localOrigin}/#rootcause`));
