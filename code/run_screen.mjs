// 제목/초록 스크리닝 채점기 (SYNERGY 벤치마크용). screen/run_screen.mjs 를 일반화한 것.
// 보내는 것은 공개 제목·초록과 공개 적격 기준뿐이다. 라벨은 보내지 않는다.
// 사용: node run_screen.mjs <jev|gateway:<model>|claude:<alias>> <출력json> --data <records.json> --criteria <criteria.json>
//       [--batch N] [--conc N] [--gap ms] [--limit N] [--subset ids.json]
// records.json: [{id, review, title, abstract, label, ...}], criteria.json: {review: "기준 문단"}
import fs from 'node:fs';
import { execFile } from 'node:child_process';
const BACKEND = process.argv[2]; const OUT = process.argv[3];
const argi = (k, d) => (process.argv.includes(k) ? process.argv[process.argv.indexOf(k) + 1] : d);
const CONC = Number(argi('--conc', 2)), GAP = Number(argi('--gap', 700)), LIMIT = Number(argi('--limit', 0)), BATCH = Number(argi('--batch', 1));
const TOKEN = fs.readFileSync('../oidc.txt', 'utf8').trim();
const DATA = JSON.parse(fs.readFileSync(argi('--data'), 'utf8'));
const CRIT = JSON.parse(fs.readFileSync(argi('--criteria'), 'utf8'));
const SUB = argi('--subset', null) ? new Set(JSON.parse(fs.readFileSync(argi('--subset'), 'utf8'))) : null;

const QUESTIONS_ALT = {
  q0: 'Based on the title and abstract, should this record be advanced to full-text screening for this review? Give the probability that it meets the eligibility criteria.',
  q1: 'Does this record potentially satisfy all of the eligibility criteria of this review, so that its full text should be retrieved and assessed?',
  q2: 'Is this study likely to be included in this systematic review?',
};
const QUESTION = QUESTIONS_ALT[argi('--q', 'q0')];
const recText = (d) => `Title: ${d.title || '(no title)'}\nAbstract: ${d.abstract || '(no abstract available; judge from the title)'}`;

let spent = 0, throttled = 0, nextSlot = 0;
const sleep = (ms) => new Promise((s) => setTimeout(s, ms));
async function pace() { const now = Date.now(); const at = Math.max(now, nextSlot); nextSlot = at + GAP; if (at > now) await sleep(at - now); }
async function backoff(r, attempt) { throttled++; const ra = Number(r.headers.get('retry-after')) || 5; const w = ra * 1000 + Math.random() * 1500; nextSlot = Math.max(nextSlot, Date.now() + w); await sleep(w); }

// Jev: 한 요청에 기록 여러 개(batch)를 넣고 기록마다 noul 질문 하나
async function askJev(recs, attempt = 1) {
  const crit = CRIT[recs[0].review];
  let state, questions;
  if (recs.length === 1) { state = crit + '\n\n' + recText(recs[0]); questions = { r1: { type: 'noul', instructions: QUESTION } }; }
  else {
    state = crit + '\n\nThe following records are independent candidates retrieved by the search; judge each one on its own.\n\n' + recs.map((d, i) => `[Record R${i + 1}]\n${recText(d)}`).join('\n\n');
    questions = {}; recs.forEach((_, i) => { questions['r' + (i + 1)] = { type: 'noul', instructions: `Consider only Record R${i + 1}. ${QUESTION}` }; });
  }
  try {
    await pace();
    const r = await fetch('https://ai-gateway.vercel.sh/typesafe/v1/systemone', { method: 'POST', headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' }, body: JSON.stringify({ model: 'typesafe-ai/jev', state, questions }), signal: AbortSignal.timeout(120000) });
    if (r.status === 429 || r.status >= 500) { if (attempt > 40) return recs.map(() => ({ error: `HTTP ${r.status} x${attempt}` })); await backoff(r, attempt); return askJev(recs, attempt + 1); }
    if (!r.ok) { const t = (await r.text()).slice(0, 150); return recs.map(() => ({ error: `HTTP ${r.status}: ${t}` })); }
    const j = await r.json(); spent += Number(j.provider_metadata?.gateway?.marketCost || 0);
    return recs.map((_, i) => ({ p: j.answers['r' + (i + 1)]?.noul, tokens: j.usage?.input_tokens, batch: recs.length }));
  } catch (e) { if (attempt <= 8) { await sleep(1500 * attempt); return askJev(recs, attempt + 1); } return recs.map(() => ({ error: String(e) })); }
}
const SYS = 'You are screening records for a systematic review. Answer with a single JSON object and nothing else: {"p": <probability 0-1 that the record meets the eligibility criteria>}. A probability of 0.2 should mean about 20% of such records turn out eligible.';
const parseP = (text) => { const m = text.match(/\{[\s\S]*\}/); const v = Number(JSON.parse(m[0]).p); if (!Number.isFinite(v) || v < 0 || v > 1) throw new Error('bad p'); return v; };
const userMsg = (d) => CRIT[d.review] + '\n\n' + recText(d) + '\n\n' + QUESTION + '\nReturn only the JSON object.';
async function askGateway(model, d, attempt = 1) {
  try {
    await pace();
    const r = await fetch('https://ai-gateway.vercel.sh/v1/chat/completions', { method: 'POST', headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' }, body: JSON.stringify({ model, temperature: 0, max_tokens: 60, messages: [{ role: 'system', content: SYS }, { role: 'user', content: userMsg(d) }] }), signal: AbortSignal.timeout(120000) });
    if (r.status === 429 || r.status >= 500) { if (attempt > 40) return { error: `HTTP ${r.status} x${attempt}` }; await backoff(r, attempt); return askGateway(model, d, attempt + 1); }
    if (!r.ok) return { error: `HTTP ${r.status}: ${(await r.text()).slice(0, 150)}` };
    const j = await r.json(); spent += Number(j.usage?.cost || 0); const text = j.choices?.[0]?.message?.content || '';
    try { return { p: parseP(text), raw: text }; } catch { if (attempt <= 3) { await sleep(1000); return askGateway(model, d, attempt + 1); } return { error: 'parse', raw: text }; }
  } catch (e) { if (attempt <= 8) { await sleep(1500 * attempt); return askGateway(model, d, attempt + 1); } return { error: String(e) }; }
}
// 토큰 로그확률 방식(OpenAI 계열): 첫 토큰의 Yes/No 확률로 P(yes)를 계산한다. 연속값이라 순위 동률이 없다.
const SYS_LP = 'You are screening records for a systematic review. Answer with a single word: Yes or No.';
const userLP = (d) => CRIT[d.review] + '\n\n' + recText(d) + '\n\nBased on the title and abstract, should this record be advanced to full-text screening for this review? Answer Yes or No.';
async function askGatewayLP(model, d, attempt = 1) {
  try {
    await pace();
    const r = await fetch('https://ai-gateway.vercel.sh/v1/chat/completions', { method: 'POST', headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, temperature: 0, max_tokens: 16, providerOptions: { gateway: { only: ['openai'] }, openai: { logprobs: true, topLogprobs: 20 } }, messages: [{ role: 'system', content: SYS_LP }, { role: 'user', content: userLP(d) }] }), signal: AbortSignal.timeout(120000) });
    if (r.status === 429 || r.status >= 500) { if (attempt > 40) return { error: `HTTP ${r.status} x${attempt}` }; await backoff(r, attempt); return askGatewayLP(model, d, attempt + 1); }
    if (!r.ok) return { error: `HTTP ${r.status}: ${(await r.text()).slice(0, 150)}` };
    const j = await r.json(); spent += Number(j.usage?.cost || 0);
    const msg = j.choices?.[0]?.message || {};
    const lps = msg.provider_metadata?.openai?.logprobs?.[0]?.[0]?.top_logprobs;
    if (!lps) { if (attempt <= 3) { await sleep(1000); return askGatewayLP(model, d, attempt + 1); } return { error: 'no logprobs', raw: msg.content }; }
    let py = 0, pn = 0;
    for (const t of lps) { const w = t.token.trim().toLowerCase(); if (w === 'yes') py += Math.exp(t.logprob); else if (w === 'no') pn += Math.exp(t.logprob); }
    if (py + pn === 0) return { error: 'no yes/no in top', raw: msg.content };
    return { p: py / (py + pn), raw: msg.content, pmass: py + pn };
  } catch (e) { if (attempt <= 8) { await sleep(1500 * attempt); return askGatewayLP(model, d, attempt + 1); } return { error: String(e) }; }
}
const env = { ...process.env }; for (const k of ['CLAUDECODE', 'CLAUDE_CODE_SESSION_ID', 'CLAUDE_CODE_CHILD_SESSION', 'CLAUDE_CODE_MESSAGING_TOKEN', 'CLAUDE_CODE_MESSAGING_SOCKET', 'CLAUDE_CODE_SSE_PORT', 'CLAUDE_PID']) delete env[k];
const CWD = '/geode3/home/u100/tjo/Quartz/tmp_claude/claude-1704385/-N-project-AiLab-jev/313fa455-92e3-4d0a-b637-fc65a2872058/scratchpad/judge';
function askClaude(alias, d, attempt = 1) {
  return new Promise((resolve) => {
    const args = ['-p', '--no-session-persistence', '--tools', '', '--setting-sources', '', '--strict-mcp-config', '--model', alias, '--system-prompt', SYS, '--output-format', 'json', userMsg(d)];
    const child = execFile('claude', args, { cwd: CWD, env, maxBuffer: 10 * 1024 * 1024, timeout: 180000 }, async (err, stdout) => {
      let j = null; try { j = JSON.parse(stdout.slice(stdout.indexOf('{'))); } catch {}
      if (!j || j.is_error || err) { if (attempt <= 6) { await sleep(Math.min(2 ** attempt * 1000, 60000)); return resolve(askClaude(alias, d, attempt + 1)); } return resolve({ error: 'cli: ' + ((j && j.result) || (err && err.message) || '').slice(0, 150) }); }
      spent += j.total_cost_usd || 0;
      try { resolve({ p: parseP(j.result || ''), raw: j.result, modelId: Object.keys(j.modelUsage || {})[0] }); } catch { if (attempt <= 3) return resolve(askClaude(alias, d, attempt + 1)); resolve({ error: 'parse', raw: j.result }); }
    });
    child.stdin.end();
  });
}

const done = fs.existsSync(OUT) ? JSON.parse(fs.readFileSync(OUT, 'utf8')).filter((d) => d.ok) : [];
const doneSet = new Set(done.map((d) => d.id));
let todo = DATA.filter((d) => !doneSet.has(d.id) && (!SUB || SUB.has(d.id)) && CRIT[d.review]); if (LIMIT) todo = todo.slice(0, LIMIT);
// 묶음: 같은 리뷰끼리 순서대로 BATCH개씩
const units = [];
if (BACKEND === 'jev' && BATCH > 1) { const byRev = {}; for (const d of todo) (byRev[d.review] ||= []).push(d); for (const arr of Object.values(byRev)) for (let i = 0; i < arr.length; i += BATCH) units.push(arr.slice(i, i + BATCH)); }
else for (const d of todo) units.push([d]);
console.log(`${BACKEND} batch=${BATCH}: ${DATA.length}건, 완료 ${done.length}, 남은 ${todo.length} (요청 ${units.length}개)`);
const out = [...done]; let i = 0, fin = 0;
const save = () => fs.writeFileSync(OUT, JSON.stringify(out));
async function worker() {
  while (i < units.length) {
    const u = units[i++];
    let answers;
    if (BACKEND === 'jev') answers = await askJev(u);
    else if (BACKEND.startsWith('gateway:')) answers = [await askGateway(BACKEND.slice(8), u[0])];
    else if (BACKEND.startsWith('gatewaylp:')) answers = [await askGatewayLP(BACKEND.slice(10), u[0])];
    else answers = [await askClaude(BACKEND.slice(7), u[0])];
    u.forEach((d, k) => { const a = answers[k] || { error: 'missing' }; out.push({ id: d.id, review: d.review, label: d.label, backend: BACKEND, ...a, ok: !a.error && Number.isFinite(a.p) }); });
    fin += u.length;
    if (fin % 50 < u.length || i === units.length) { save(); console.log(`  ${fin}/${todo.length}  비용 $${spent.toFixed(4)}  혼잡 ${throttled}  실패 ${out.filter((o) => !o.ok).length}`); }
  }
}
await Promise.all(Array.from({ length: CONC }, worker)); save();
console.log(`완료. 비용 $${spent.toFixed(4)}, 혼잡 ${throttled}, 실패 ${out.filter((o) => !o.ok).length}`);
