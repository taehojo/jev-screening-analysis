// npj AN-0001-15 stage B: copy of synergy/run_screen.mjs (Jev backend only) for one interleaved run of two criteria versions.
// Differences from the original (see README.md): (1) two criteria files (abstract-derived, protocol-derived) scored in one run;
// units are built per review in data-file order, ten-record batches within the review as in the original, and the two conditions
// of each review are queued next to each other (abstract first in even-numbered reviews, protocol first in odd-numbered ones);
// (2) outputs go to this folder, one file per condition, without the label field; (3) per request the time, the model field and
// the final provider of the gateway response, the record count and the reported cost are appended to api_spend.csv (O_APPEND,
// like a shell >>) and to a local batch log; (4) a spending cap: no new request is started once the reported spend reaches CAP.
// The question, the prompt template, the batch size, the pacing (gap, concurrency) and the 429/5xx handling with retry-after are
// unchanged. Only public titles, abstracts and published criteria are sent; no labels. The token is read from oidc.txt and never written.
// Usage: node an15_run_screen_paired.mjs <records.json> <abstract_criteria.json> <protocol_criteria.json> <reviews,comma> <cap_usd> [--conc 3] [--gap 1900] [--batch 10]
import fs from 'node:fs';
const [RECS, CA, CP, REVS, CAPS] = process.argv.slice(2, 7);
const argi = (k, d) => (process.argv.includes(k) ? process.argv[process.argv.indexOf(k) + 1] : d);
const CONC = Number(argi('--conc', 3)), GAP = Number(argi('--gap', 1900)), BATCH = Number(argi('--batch', 10)); const CAP = Number(CAPS);
const TOKEN = fs.readFileSync('/N/project/AiLab/jev/oidc.txt', 'utf8').trim();
const SPEND = '/N/project/AiLab/jev/review_pipeline_npj/api_spend.csv';
const DATA = JSON.parse(fs.readFileSync(RECS, 'utf8')); const CRIT = { abstract: JSON.parse(fs.readFileSync(CA, 'utf8')), protocol: JSON.parse(fs.readFileSync(CP, 'utf8')) };
const REVIEWS = REVS.split(',');
const QUESTION = 'Based on the title and abstract, should this record be advanced to full-text screening for this review? Give the probability that it meets the eligibility criteria.';
const recText = (d) => `Title: ${d.title || '(no title)'}\nAbstract: ${d.abstract || '(no abstract available; judge from the title)'}`;
let spent = 0, throttled = 0, nextSlot = 0;
const sleep = (ms) => new Promise((s) => setTimeout(s, ms));
async function pace() { const now = Date.now(); const at = Math.max(now, nextSlot); nextSlot = at + GAP; if (at > now) await sleep(at - now); }
async function backoff(r, attempt) { throttled++; const ra = Number(r.headers.get('retry-after')) || 5; const w = ra * 1000 + Math.random() * 1500; nextSlot = Math.max(nextSlot, Date.now() + w); await sleep(w); }
const OUTF = { abstract: 'jev_clef_abstract_criteria_new.json', protocol: 'jev_clef_protocol_criteria_new.json' };
const BLOG = 'an15_batches.tsv';
if (!fs.existsSync(BLOG)) fs.appendFileSync(BLOG, 'utc\tcondition\treview\tn_records\tstatus\tmodel\tfinal_provider\tinput_tokens\tcost_usd\tattempt\n');
async function askJev(cond, recs, attempt = 1) {
  const crit = CRIT[cond][recs[0].review];
  let state, questions;
  if (recs.length === 1) { state = crit + '\n\n' + recText(recs[0]); questions = { r1: { type: 'noul', instructions: QUESTION } }; }
  else {
    state = crit + '\n\nThe following records are independent candidates retrieved by the search; judge each one on its own.\n\n' + recs.map((d, i) => `[Record R${i + 1}]\n${recText(d)}`).join('\n\n');
    questions = {}; recs.forEach((_, i) => { questions['r' + (i + 1)] = { type: 'noul', instructions: `Consider only Record R${i + 1}. ${QUESTION}` }; });
  }
  try {
    await pace();
    const r = await fetch('https://ai-gateway.vercel.sh/typesafe/v1/systemone', { method: 'POST', headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' }, body: JSON.stringify({ model: 'typesafe-ai/jev', state, questions }), signal: AbortSignal.timeout(120000) });
    if (r.status === 429 || r.status >= 500) { if (attempt > 40) return recs.map(() => ({ error: `HTTP ${r.status} x${attempt}` })); await backoff(r, attempt); return askJev(cond, recs, attempt + 1); }
    if (!r.ok) { const t = (await r.text()).slice(0, 150); return recs.map(() => ({ error: `HTTP ${r.status}: ${t}` })); }
    const j = await r.json(); const cost = Number(j.provider_metadata?.gateway?.marketCost || 0); spent += cost;
    const model = j.model || ''; const prov = j.provider_metadata?.gateway?.routing?.finalProvider || '';
    const utc = new Date().toISOString();
    fs.appendFileSync(SPEND, `${utc},1,AN-0001-15,vercel-ai-gateway/${prov},${model},${recs.length},${cost.toFixed(8)},response provider_metadata.gateway.marketCost (stage B ${cond} criteria; review ${recs[0].review})\n`);
    fs.appendFileSync(BLOG, `${utc}\t${cond}\t${recs[0].review}\t${recs.length}\t${r.status}\t${model}\t${prov}\t${j.usage?.input_tokens}\t${cost}\t${attempt}\n`);
    return recs.map((_, i) => ({ p: j.answers['r' + (i + 1)]?.noul, tokens: j.usage?.input_tokens, batch: recs.length, model, provider: prov, utc }));
  } catch (e) { if (attempt <= 8) { await sleep(1500 * attempt); return askJev(cond, recs, attempt + 1); } return recs.map(() => ({ error: String(e) })); }
}
const done = {}; const out = {};
for (const c of ['abstract', 'protocol']) { out[c] = fs.existsSync(OUTF[c]) ? JSON.parse(fs.readFileSync(OUTF[c], 'utf8')).filter((d) => d.ok) : []; done[c] = new Set(out[c].map((d) => d.id)); }
const units = [];
REVIEWS.forEach((rev, ri) => {
  const arr = DATA.filter((d) => d.review === rev);
  const order = ri % 2 === 0 ? ['abstract', 'protocol'] : ['protocol', 'abstract'];
  for (const c of order) { const todo = arr.filter((d) => !done[c].has(d.id)); for (let i = 0; i < todo.length; i += BATCH) units.push({ c, recs: todo.slice(i, i + BATCH) }); }
});
console.log(`reviews ${REVIEWS.length}, requests ${units.length}, records ${units.reduce((s, u) => s + u.recs.length, 0)}, done abstract ${done.abstract.size} protocol ${done.protocol.size}, cap $${CAP}`);
let i = 0, fin = 0, capped = false;
const save = () => { for (const c of ['abstract', 'protocol']) fs.writeFileSync(OUTF[c], JSON.stringify(out[c])); };
async function worker() {
  while (i < units.length) {
    if (spent >= CAP) { capped = true; break; }
    const u = units[i++]; const answers = await askJev(u.c, u.recs);
    u.recs.forEach((d, k) => { const a = answers[k] || { error: 'missing' }; out[u.c].push({ id: d.id, review: d.review, condition: u.c, backend: 'jev', ...a, ok: !a.error && Number.isFinite(a.p) }); });
    fin += u.recs.length;
    if (fin % 500 < u.recs.length || i === units.length) { save(); console.log(`  ${new Date().toISOString()} ${fin} records  cost $${spent.toFixed(6)}  throttled ${throttled}  failed ${out.abstract.concat(out.protocol).filter((o) => !o.ok).length}`); }
  }
}
await Promise.all(Array.from({ length: CONC }, worker)); save();
console.log(`finished ${new Date().toISOString()}. cost $${spent.toFixed(6)}, throttled ${throttled}, failed ${out.abstract.concat(out.protocol).filter((o) => !o.ok).length}, capped ${capped}`);
