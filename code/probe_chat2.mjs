import fs from 'node:fs';
const T = fs.readFileSync('../oidc.txt', 'utf8').trim();
for (let i = 0; i < 3; i++) {
  const t0 = Date.now();
  const r = await fetch('https://ai-gateway.vercel.sh/v1/chat/completions', { method: 'POST', headers: { Authorization: `Bearer ${T}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'openai/gpt-4o-mini', max_tokens: 16, messages: [{ role: 'user', content: 'Reply OK' }] }), signal: AbortSignal.timeout(60000) });
  const txt = await r.text();
  console.log(r.status, (Date.now() - t0) + 'ms', 'retry-after', r.headers.get('retry-after'), 'x-ratelimit-limit-requests', r.headers.get('x-ratelimit-limit-requests'), 'remaining', r.headers.get('x-ratelimit-remaining-requests'), 'reset', r.headers.get('x-ratelimit-reset-requests'), txt.slice(0, 160).replace(/\s+/g, ' '));
}
