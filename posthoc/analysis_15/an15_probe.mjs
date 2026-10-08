// AN-0001-15: one probe request to the Jev endpoint through the Vercel AI Gateway, recording the full response body and headers
// (presence or absence of a model identifier or build string). The request carries a neutral public-domain sentence only:
// no manuscript text, no study record, no ADSP data (INTEGRITY_RULES I-14). The token is read from oidc.txt and never written.
// Usage: node an15_probe.mjs   (run from this folder)
import fs from 'node:fs';
const TOKEN = fs.readFileSync('/N/project/AiLab/jev/oidc.txt', 'utf8').trim();
const H = { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' };
const ts = () => new Date().toISOString();
const headersObj = (r) => { const o = {}; r.headers.forEach((v, k) => { o[k] = v; }); return o; };
const redact = (o) => { const c = { ...o }; for (const k of Object.keys(c)) if (/authorization|cookie|set-cookie/i.test(k)) c[k] = '[not recorded]'; return c; };

const record = { started_utc: ts(), endpoint: 'https://ai-gateway.vercel.sh/typesafe/v1/systemone', model: 'typesafe-ai/jev', steps: [] };
// 1. credits before (free GET)
let r = await fetch('https://ai-gateway.vercel.sh/v1/credits', { headers: { Authorization: `Bearer ${TOKEN}` } });
record.steps.push({ step: 'credits_before', utc: ts(), status: r.status, body: await r.json().catch(async () => await r.text()) });
// 2. the probe: a neutral public-domain sentence and one probability question
const body = {
  model: 'typesafe-ai/jev',
  state: 'Water boils at 100 degrees Celsius at standard atmospheric pressure.',
  questions: { q1: { type: 'noul', instructions: 'Is this statement about the boiling point of water?' } },
};
record.request_body = body;
const t0 = Date.now();
r = await fetch(record.endpoint, { method: 'POST', headers: H, body: JSON.stringify(body), signal: AbortSignal.timeout(120000) });
const latency_ms = Date.now() - t0;
const text = await r.text(); let json = null; try { json = JSON.parse(text); } catch {}
record.steps.push({ step: 'probe', utc: ts(), status: r.status, status_text: r.statusText, latency_ms, response_headers: redact(headersObj(r)), response_body: json ?? text });
// 3. credits after (free GET)
r = await fetch('https://ai-gateway.vercel.sh/v1/credits', { headers: { Authorization: `Bearer ${TOKEN}` } });
record.steps.push({ step: 'credits_after', utc: ts(), status: r.status, body: await r.json().catch(async () => await r.text()) });
// 4. public model listing entry for the model (no authorisation header)
r = await fetch('https://ai-gateway.vercel.sh/v1/models');
const models = await r.json().catch(() => null);
const entry = models && models.data ? models.data.find((m) => m.id === 'typesafe-ai/jev') : null;
record.steps.push({ step: 'public_models_listing', utc: ts(), status: r.status, n_models: models && models.data ? models.data.length : null, jev_entry: entry });
record.finished_utc = ts();
// summary of what the response carries
const probe = record.steps.find((s) => s.step === 'probe');
const keysTop = probe.response_body && typeof probe.response_body === 'object' ? Object.keys(probe.response_body) : null;
record.summary = {
  http_status: probe.status,
  top_level_keys: keysTop,
  gateway_market_cost_usd: probe.response_body?.provider_metadata?.gateway?.marketCost ?? null,
  usage: probe.response_body?.usage ?? null,
  answer: probe.response_body?.answers ?? null,
  model_identifier_fields_in_body: keysTop ? keysTop.filter((k) => /model|version|build|id/i.test(k)) : null,
  header_names: Object.keys(probe.response_headers),
  header_names_mentioning_model_or_version: Object.keys(probe.response_headers).filter((k) => /model|version|build/i.test(k)),
};
fs.writeFileSync('probe_record.json', JSON.stringify(record, null, 1));
console.log(JSON.stringify(record.summary, null, 1));
console.log('credits before/after:', JSON.stringify(record.steps[0].body), JSON.stringify(record.steps[2].body));
