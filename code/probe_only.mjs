import fs from 'node:fs';
const T = fs.readFileSync('../oidc.txt', 'utf8').trim();
for (const po of [{ gateway: { only: ['openai'] } }, { gateway: { order: ['openai'] } }]) {
  const r = await fetch('https://ai-gateway.vercel.sh/v1/chat/completions', { method: 'POST', headers: { Authorization: `Bearer ${T}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'openai/gpt-4o-mini', temperature: 0, max_tokens: 16, providerOptions: { ...po, openai: { logprobs: true, topLogprobs: 5 } }, messages: [{ role: 'user', content: 'Is Paris the capital of France? Answer Yes or No.' }] }), signal: AbortSignal.timeout(60000) });
  const j = await r.json().catch(() => ({}));
  const m = j.choices?.[0]?.message || {};
  console.log(JSON.stringify(po), r.status, 'finalProvider', m.provider_metadata?.gateway?.routing?.finalProvider, 'has logprobs', !!m.provider_metadata?.openai?.logprobs, j.error?.message?.slice(0, 150) || '');
}
