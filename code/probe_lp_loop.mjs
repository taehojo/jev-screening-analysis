import fs from 'node:fs';
const T = fs.readFileSync('../oidc.txt', 'utf8').trim();
const C = JSON.parse(fs.readFileSync('criteria.json', 'utf8'));
const R = JSON.parse(fs.readFileSync('test_records.json', 'utf8')).slice(0, 2);
for (const d of R) {
  const t0 = Date.now();
  const r = await fetch('https://ai-gateway.vercel.sh/v1/chat/completions', { method: 'POST', headers: { Authorization: `Bearer ${T}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'openai/gpt-4o-mini', temperature: 0, max_tokens: 16, providerOptions: { openai: { logprobs: true, topLogprobs: 20 } }, messages: [{ role: 'system', content: 'Answer Yes or No.' }, { role: 'user', content: C[d.review] + '\n\nTitle: ' + d.title + '\nAbstract: ' + d.abstract + '\n\nShould this record be advanced to full-text screening? Answer Yes or No.' }] }), signal: AbortSignal.timeout(120000) });
  const t = await r.text();
  console.log(r.status, Date.now() - t0, 'ms', 'retry-after', r.headers.get('retry-after'), t.slice(0, 400).replace(/\s+/g, ' '));
  await new Promise((s) => setTimeout(s, 4000));
}
