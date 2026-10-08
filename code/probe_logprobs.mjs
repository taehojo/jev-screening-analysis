import fs from 'node:fs';
const T = fs.readFileSync('../oidc.txt', 'utf8').trim();
for (const model of ['openai/gpt-4o-mini', 'deepseek/deepseek-v3.1', 'openai/gpt-4o']) {
  const r = await fetch('https://ai-gateway.vercel.sh/v1/chat/completions', { method: 'POST', headers: { Authorization: `Bearer ${T}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, temperature: 0, max_tokens: 1, logprobs: true, top_logprobs: 5, messages: [{ role: 'user', content: 'Is Paris the capital of France? Answer Yes or No.' }] }), signal: AbortSignal.timeout(60000) });
  const t = await r.text();
  let lp = null; try { const j = JSON.parse(t); lp = j.choices?.[0]?.logprobs; console.log(model, r.status, 'content:', j.choices?.[0]?.message?.content, '| logprobs:', JSON.stringify(lp)?.slice(0, 300)); } catch { console.log(model, r.status, t.slice(0, 200)); }
}
