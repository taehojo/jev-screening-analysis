import fs from 'node:fs';
const T = fs.readFileSync('../oidc.txt', 'utf8').trim();
const tries = [
  { model: 'openai/gpt-4o-mini', max_tokens: 16, logprobs: true, top_logprobs: 5 },
  { model: 'openai/gpt-4o-mini', max_tokens: 16, providerOptions: { openai: { logprobs: 5 } } },
  { model: 'openai/gpt-4o-mini', max_tokens: 16, providerOptions: { openai: { logprobs: true, topLogprobs: 5 } } },
];
for (const extra of tries) {
  const r = await fetch('https://ai-gateway.vercel.sh/v1/chat/completions', { method: 'POST', headers: { Authorization: `Bearer ${T}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ temperature: 0, messages: [{ role: 'user', content: 'Is Paris the capital of France? Answer Yes or No.' }], ...extra }), signal: AbortSignal.timeout(60000) });
  const t = await r.text();
  console.log(JSON.stringify(extra).slice(0, 90), '->', r.status, t.slice(0, 700).replace(/\s+/g, ' '));
  console.log();
}
