# AN-0001-12: cost claims with price sources, token counts, single-record cost, ratios, throughput from run logs.
import json, os, re, html, numpy as np, pandas as pd
from datetime import datetime
S = '/N/project/AiLab/jev/synergy/'; SC = '/N/project/AiLab/jev/screen/'; OUT = os.path.dirname(os.path.abspath(__file__)); PS = OUT + '/pricing_sources/'
res = {}
# 1. prices (gateway listing fetched 2026-09-26; vendor pages as cross-checks)
gw = json.load(open(PS + 'vercel_ai_gateway_v1_models.json'))['data']
def gwprice(mid):
    it = [x for x in gw if x['id'] == mid][0]; p = it['pricing']
    return {'id': mid, 'name': it.get('name'), 'input_usd_per_M': float(p['input']) * 1e6, 'output_usd_per_M': float(p['output']) * 1e6, 'input_cache_read_usd_per_M': (float(p['input_cache_read']) * 1e6 if p.get('input_cache_read') else None),
            'input_cache_write_usd_per_M': (float(p['input_cache_write']) * 1e6 if p.get('input_cache_write') else None), 'varies_by_provider': p.get('varies_by_provider', False), 'released_unix': it.get('released'), 'released_date_utc': datetime.utcfromtimestamp(it['released']).strftime('%Y-%m-%d') if it.get('released') else None}
res['prices_gateway_listing'] = {k: gwprice(v) for k, v in [('jev', 'typesafe-ai/jev'), ('gpt4omini', 'openai/gpt-4o-mini'), ('deepseek_v31', 'deepseek/deepseek-v3.1'), ('claude_opus_55', 'anthropic/claude-opus-5.5')]}
res['prices_gateway_listing']['source'] = 'https://ai-gateway.vercel.sh/v1/models (public listing) fetched 2026-09-26 17:52 EDT; copy in pricing_sources/'
def txt(f):
    s = open(f, encoding='utf-8', errors='ignore').read(); s = re.sub(r'<script[\s\S]*?</script>|<style[\s\S]*?</style>', '', s); s = re.sub(r'<[^>]+>', ' ', s); s = html.unescape(s); return re.sub(r'\s+', ' ', s)
def snip(t, pat, w=220, n=3):
    return [t[max(0, m.start() - w):m.start() + w] for m in list(re.finditer(pat, t))[:n]]
ta = txt(PS + 'platform_claude_com_pricing.html'); to = txt(PS + 'developers_openai_com_pricing.html'); tt = txt(PS + 'typesafe_blog_introducing_jev.html'); td = txt(PS + 'api-docs_deepseek_com_pricing.html')
raw_openai = open(PS + 'developers_openai_com_pricing.html', encoding='utf-8', errors='ignore').read()
m = re.search(r'gpt-4o-mini&quot;\],\[0,([0-9.]+)\],\[0,([0-9.]+)\],\[0,([0-9.]+)\]', raw_openai)
res['prices_vendor_pages'] = {
    'anthropic_claude_opus_55': {'snippet': snip(ta, r'Claude Opus 5\.5 For long-running', 60, 1), 'reading': 'Claude Opus 5.5: $4 / MTok input, $20 / MTok output, $5 / MTok 5-minute cache write, $8 / MTok 1-hour cache write, $0.20 / MTok cache read (columns as on the page)', 'batch_snippets': snip(ta, r'(?i)batch api', 200, 3), 'cache_snippets': snip(ta, r'(?i)cache hit costs', 200, 2), 'url': 'https://platform.claude.com/docs/en/about-claude/pricing', 'fetched': '2026-09-26 17:52 EDT'},
    'openai_gpt4omini': {'table_row_input_cached_output_usd_per_M': [float(m.group(1)), float(m.group(2)), float(m.group(3))] if m else None, 'batch_snippets': snip(to, r'(?i)batch api', 200, 3), 'url': 'https://developers.openai.com/api/docs/pricing (redirect from platform.openai.com/docs/pricing)', 'fetched': '2026-09-26 17:52 EDT'},
    'deepseek': {'note': 'The vendor pricing page fetched on 2026-09-26 lists deepseek-flash and deepseek-v4-pro only; DeepSeek-V3.1 is no longer listed, so the gateway listing price is used.', 'snippet': snip(td, r'MODEL deepseek', 100, 1), 'url': 'https://api-docs.deepseek.com/quick_start/pricing'},
    'typesafe_jev': {'price_snippets': snip(tt, r'\$0\.042', 150, 2), 'speed_snippets': snip(tt, r'(?i)end-to-end response time', 250, 2), 'orders_snippets': snip(tt, r'(?i)two orders of magnitude', 200, 1), 'pricing_sustainability_snippet': snip(tt, r'(?i)We make our pricing transparent', 250, 1), 'url': 'https://typesafe.ai/blog/introducing-system-one-models-and-jev', 'fetched': '2026-09-26 17:52 EDT'}}
# 2. costs from the run logs (gateway-reported for Jev, GPT-4o-mini, DeepSeek; interface-reported for Claude)
def done_costs(f):
    path = f if f.startswith('/') else S + f
    return [float(x) for x in re.findall(r'완료\. 비용 \$([0-9.]+)', open(path).read())]
def n_ok(f):
    d = json.load(open(f)); return sum(1 for x in d if x.get('ok'))
runs = {}
for key, log, out, n in [('jev_development', 'run_jev_dev.log', S + 'jev_dev2000.json', None), ('jev_held_out', 'run_jev_test.log', S + 'jev_test.json', None), ('jev_clef', 'run_jev_clef.log', S + 'jev_clef.json', None),
                         ('jev_robust_q1', 'run_jev_q1.log', S + 'jev_robust_q1.json', None), ('jev_robust_q2', 'run_jev_q2.log', S + 'jev_robust_q2.json', None), ('jev_robust_titleonly', 'run_jev_titleonly.log', S + 'jev_robust_titleonly.json', None), ('jev_robust_mismatch', 'run_jev_mismatch.log', S + 'jev_robust_mismatch.json', None),
                         ('jev_p2_single_record', 'run_p2_single.log', S + 'jev_testcmp_single.json', None), ('jev_p2_batch10_subset', 'run_p2_b10.log', S + 'jev_testcmp_b10.json', None),
                         ('gpt4omini_lp_resumed_run', 'run_test_gpt4omini.log', S + 'test_cmp_gpt4omini_lp.json', 1957), ('deepseek', 'run_test_deepseek.log', S + 'test_cmp_deepseek.json', None), ('claude_opus_cli', 'run_test_claude.log', S + 'test_cmp_claude-opus.json', None),
                         ('pilot_jev_single', SC + 'run_jev.log', SC + 'screen_jev.json', None), ('pilot_claude_opus_cli', SC + 'run_claude-opus.log', SC + 'screen_claude-opus.json', None)]:
    c = done_costs(log)
    runs[key] = {'log': log, 'completed_run_costs_usd': c, 'cost_first_pass_usd': c[0] if c else None, 'cost_all_passes_usd': round(sum(c), 4), 'n_records_ok_in_output': n_ok(out), 'n_records_in_first_pass': n}
res['run_costs'] = runs
res['run_costs']['gpt4omini_lp_first_run_note'] = 'The first gpt-4o-mini run (Sept 24, 19:00, progress.log) reached 50/2002 at $0.0087 with 5 failures and was restarted at 19:02 with 45 records kept; its cost is only in progress.log. Total gateway cost for 2002 records = 0.2133 + 0.0087 = 0.2220 USD (the 0.0087 covers 45 successful and 5 failed attempts).'
gpt_total = 0.2133 + 0.0087
res['cost_per_1000'] = {'jev_development_batch10': round(1000 * runs['jev_development']['cost_first_pass_usd'] / runs['jev_development']['n_records_ok_in_output'], 4),
                        'jev_held_out_batch10': round(1000 * runs['jev_held_out']['cost_first_pass_usd'] / runs['jev_held_out']['n_records_ok_in_output'], 4),
                        'jev_clef_batch10': round(1000 * runs['jev_clef']['cost_first_pass_usd'] / runs['jev_clef']['n_records_ok_in_output'], 4),
                        'jev_p2_batch10_subset': round(1000 * runs['jev_p2_batch10_subset']['cost_first_pass_usd'] / 2002, 4), 'jev_p2_single_record': round(1000 * runs['jev_p2_single_record']['cost_first_pass_usd'] / 2002, 4),
                        'jev_pilot_single_record': round(1000 * runs['pilot_jev_single']['cost_first_pass_usd'] / 1572, 4),
                        'gpt4omini_lp_single_record': round(1000 * gpt_total / 2002, 4), 'deepseek_single_record': round(1000 * runs['deepseek']['cost_first_pass_usd'] / 2002, 4),
                        'claude_opus_cli_per_2002_attempted': round(1000 * runs['claude_opus_cli']['cost_first_pass_usd'] / 2002, 4), 'claude_opus_cli_per_1999_scored': round(1000 * runs['claude_opus_cli']['cost_first_pass_usd'] / 1999, 4),
                        'claude_opus_cli_pilot_1572': round(1000 * runs['pilot_claude_opus_cli']['cost_first_pass_usd'] / 1572, 4)}
c = res['cost_per_1000']
res['cost_ratios'] = {'basis': 'Jev held-out batched run (0.0201 per 1000) versus each comparator on the 2002-record subset; second row uses the Jev single-record cost from the P2 run',
                      'jev_batch10_vs': {'gpt4omini_lp': round(c['gpt4omini_lp_single_record'] / c['jev_held_out_batch10'], 1), 'deepseek': round(c['deepseek_single_record'] / c['jev_held_out_batch10'], 1), 'claude_opus_cli_2002': round(c['claude_opus_cli_per_2002_attempted'] / c['jev_held_out_batch10'], 0), 'claude_opus_cli_1999': round(c['claude_opus_cli_per_1999_scored'] / c['jev_held_out_batch10'], 0)},
                      'jev_single_vs': {'gpt4omini_lp': round(c['gpt4omini_lp_single_record'] / c['jev_p2_single_record'], 1), 'deepseek': round(c['deepseek_single_record'] / c['jev_p2_single_record'], 1), 'claude_opus_cli_2002': round(c['claude_opus_cli_per_2002_attempted'] / c['jev_p2_single_record'], 0)},
                      'manuscript_v0_values': {'jev': 0.02, 'gpt4omini': 0.11, 'deepseek': 0.16, 'claude': 14.65, 'ratio_stated': 'about 1/700th'}}
# 3. Jev token counts from the output files (usage.input_tokens per request; each record of a request carries the request value)
def jev_tokens(f, nrec=None):
    d = [x for x in json.load(open(f)) if x.get('ok') and x.get('tokens') is not None]; b = np.array([x.get('batch', 1) for x in d]); t = np.array([x['tokens'] for x in d], float)
    per_rec = t / b; total = float(per_rec.sum())
    return {'n_records': len(d), 'mean_input_tokens_per_request': round(float(t.mean()), 1), 'mean_input_tokens_per_record': round(float(per_rec.mean()), 1), 'median_input_tokens_per_record': round(float(np.median(per_rec)), 1), 'total_input_tokens': int(round(total)), 'batch_sizes': {str(k): int(v) for k, v in zip(*np.unique(b, return_counts=True))},
            'cost_at_0.042_per_M_usd': round(total * 0.042e-6, 4)}
res['jev_tokens'] = {'development_batch10': jev_tokens(S + 'jev_dev2000.json'), 'held_out_batch10': jev_tokens(S + 'jev_test.json'), 'clef_batch10': jev_tokens(S + 'jev_clef.json'), 'p2_single_record': jev_tokens(S + 'jev_testcmp_single.json'), 'p2_batch10_subset': jev_tokens(S + 'jev_testcmp_b10.json'),
                     'pilot_single_record': jev_tokens(SC + 'screen_jev.json'), 'pilot_batch10': jev_tokens(S + 'dhl_jev_b10.json'), 'robust_q1_batch10': jev_tokens(S + 'jev_robust_q1.json'), 'robust_titleonly_batch10': jev_tokens(S + 'jev_robust_titleonly.json')}
res['jev_tokens']['note'] = 'usage.input_tokens is the gateway-reported input token count of the request; output tokens are free and not reported. cost_at_0.042_per_M is the arithmetic check against the gateway-reported marketCost.'
# 4. prompt lengths for the LLM comparison (exact characters and words from the code paths in run_screen.mjs), with a chars/4 approximation
R = json.load(open(S + 'test_records.json')); ids = set(json.load(open(S + 'test_cmp_ids.json'))); CRIT = json.load(open(S + 'criteria.json'))
Q = 'Based on the title and abstract, should this record be advanced to full-text screening for this review? Give the probability that it meets the eligibility criteria.'
SYS = 'You are screening records for a systematic review. Answer with a single JSON object and nothing else: {"p": <probability 0-1 that the record meets the eligibility criteria>}. A probability of 0.2 should mean about 20% of such records turn out eligible.'
SYS_LP = 'You are screening records for a systematic review. Answer with a single word: Yes or No.'
rec = lambda d: f"Title: {d['title'] or '(no title)'}\nAbstract: {d['abstract'] or '(no abstract available; judge from the title)'}"
sub = [r for r in R if r['id'] in ids]
L = {'verbalised_user_msg': [], 'lp_user_msg': [], 'jev_single_state': [], 'criteria_block': [], 'record_block': []}
for d in sub:
    crit = CRIT[d['review']]; L['criteria_block'].append(len(crit)); L['record_block'].append(len(rec(d)))
    L['verbalised_user_msg'].append(len(crit + '\n\n' + rec(d) + '\n\n' + Q + '\nReturn only the JSON object.'))
    L['lp_user_msg'].append(len(crit + '\n\n' + rec(d) + '\n\nBased on the title and abstract, should this record be advanced to full-text screening for this review? Answer Yes or No.'))
    L['jev_single_state'].append(len(crit + '\n\n' + rec(d)))
# Jev batched state reconstructed as in run_screen.mjs (consecutive records of the same review in file order, ten per request) over the full held-out set
by = {}
for r in R: by.setdefault(r['review'], []).append(r)
tot_chars = 0; nreq = 0
for k, arr in by.items():
    for i in range(0, len(arr), 10):
        u = arr[i:i + 10]; crit = CRIT[k]
        if len(u) == 1: st = crit + '\n\n' + rec(u[0])
        else: st = crit + '\n\nThe following records are independent candidates retrieved by the search; judge each one on its own.\n\n' + '\n\n'.join(f'[Record R{j + 1}]\n{rec(d)}' for j, d in enumerate(u))
        tot_chars += len(st); nreq += 1
jt = res['jev_tokens']['held_out_batch10']
res['prompt_lengths'] = {'subset_n': len(sub), 'chars_mean': {k: round(float(np.mean(v)), 1) for k, v in L.items()}, 'chars_median': {k: float(np.median(v)) for k, v in L.items()}, 'words_mean_verbalised_user_msg': round(float(np.mean([len(x.split()) for x in [CRIT[d['review']] + ' ' + rec(d) + ' ' + Q for d in sub]])), 1),
                         'system_prompt_chars': {'verbalised': len(SYS), 'log_probability': len(SYS_LP)}, 'approx_tokens_chars_div_4': {'verbalised_total_per_record': round((np.mean(L['verbalised_user_msg']) + len(SYS)) / 4, 0), 'lp_total_per_record': round((np.mean(L['lp_user_msg']) + len(SYS_LP)) / 4, 0), 'jev_single_state': round(np.mean(L['jev_single_state']) / 4, 0)},
                         'jev_held_out_batched_state_chars_total': tot_chars, 'jev_held_out_requests_reconstructed': nreq, 'jev_held_out_chars_per_record': round(tot_chars / len(R), 1), 'jev_chars_per_reported_token_held_out': round(tot_chars / jt['total_input_tokens'], 2),
                         'note': 'Characters and words are exact for the texts sent; token counts for the chat models were not recorded by the harness, so chars/4 is an approximation and the implied counts below are derived from cost at list price.'}
P = res['prices_gateway_listing']
gpt_per = gpt_total / 2002; ds_per = runs['deepseek']['cost_first_pass_usd'] / 2002; cl_per = runs['claude_opus_cli']['cost_first_pass_usd'] / 1999
res['implied_tokens_from_cost'] = {'gpt4omini_lp': {'cost_per_record_usd': round(gpt_per, 6), 'assumed_output_tokens': 1, 'implied_input_tokens': round((gpt_per - 1 * P['gpt4omini']['output_usd_per_M'] / 1e6) / (P['gpt4omini']['input_usd_per_M'] / 1e6), 0)},
                                   'deepseek': {'cost_per_record_usd': round(ds_per, 6), 'assumed_output_tokens': 8, 'implied_input_tokens_at_gateway_list_price': round((ds_per - 8 * P['deepseek_v31']['output_usd_per_M'] / 1e6) / (P['deepseek_v31']['input_usd_per_M'] / 1e6), 0), 'caveat': 'price varies by provider on the gateway'},
                                   'claude_opus_cli': {'cost_per_scored_record_usd': round(cl_per, 6), 'assumed_output_tokens': 10, 'implied_input_token_equivalents_at_4_per_M': round((cl_per - 10 * 20e-6) / 4e-6, 0), 'implied_at_5_per_M_cache_write': round((cl_per - 10 * 20e-6) / 5e-6, 0),
                                                       'caveat': 'The command-line interface reports a total cost per call; how it is composed (input, cache writes, hidden harness context) is not itemised in our record. An earlier run with the same interface settings measured a fixed context of 577 input tokens (HANDOVER.md section 11), which is far below the implied count, so the interface figure should not be read as an API invoice for the prompt text alone.'}}
# 5. throughput from timestamps
def mins(a, b): return round((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() / 60, 1)
T = [('jev_development_batch10', 52957, '2026-09-24T18:17:47', 'run_jev_dev.log creation time as verified in PREREG.md (entry of 2026-09-25 13:14)', '2026-09-25T00:48:48', 'jev_dev2000.json and run_jev_dev.log modification time (the retry pass at the end had nothing left to score)'),
     ('jev_held_out_batch10', 33001, '2026-09-25T00:48:48', 'chain_jev.sh starts the held-out run immediately after the development retry pass', '2026-09-25T03:29:32', 'jev_test.json modification time'),
     ('jev_robust_q1_batch10', 5550, '2026-09-25T03:29:32', 'chain_jev.sh sequence', '2026-09-25T03:52:49', 'jev_robust_q1.json modification time'),
     ('jev_robust_q2_batch10', 5550, '2026-09-25T03:52:49', 'chain_jev.sh sequence', '2026-09-25T04:14:28', 'jev_robust_q2.json modification time'),
     ('jev_robust_titleonly_batch10', 5550, '2026-09-25T04:14:28', 'chain2_jev.sh starts when chain_jev.log says done (04:14:28)', '2026-09-25T04:37:44', 'jev_robust_titleonly.json modification time'),
     ('jev_robust_mismatch_batch10', 5550, '2026-09-25T04:37:44', 'chain2_jev.sh sequence', '2026-09-25T05:01:25', 'jev_robust_mismatch.json modification time'),
     ('jev_clef_batch10', 82418, '2026-09-25T10:39:29', 'run_p2.done modification time; chain_clef.sh starts the CLEF run when this file appears', '2026-09-25T17:03:30', 'jev_clef.json modification time (retry pass had nothing left)'),
     ('claude_opus_cli_subset', 2002, '2026-09-24T18:30:00', 'progress.log: 0 done at 18:30, 50 done at 18:31 (2-minute polling; start uncertain by about one minute)', '2026-09-24T19:15:10', 'run_test_claude.log modification time (main pass; the retry of 3 refused records ran later)'),
     ('deepseek_subset', 2002, '2026-09-24T18:56:00', 'progress.log: run present with 0 done at 18:56, 50 done at 19:04 (start uncertain by up to 8 minutes)', '2026-09-24T23:57:18', 'test_cmp_deepseek.json modification time'),
     ('gpt4omini_lp_subset_resumed_run', 1957, '2026-09-24T19:02:00', 'progress.log: resumed run present with 45 done at 19:02, 50/1957 at 19:22', '2026-09-25T04:50:20', 'test_cmp_gpt4omini_lp.json modification time'),
     ('jev_p2_batch10_subset', 2002, None, 'start not recorded (no chain script; run_p2_single.log ended 10:28:38 and the batched run ended 10:39:29; if sequential, about 11 minutes)', '2026-09-25T10:39:29', 'run_p2_b10.log modification time')]
thr = []
for name, n, s, ssrc, e, esrc in T:
    row = {'run': name, 'n_records': n, 'start': s, 'start_source': ssrc, 'end': e, 'end_source': esrc}
    if s: row['minutes'] = mins(s, e); row['records_per_minute'] = round(n / row['minutes'], 1); row['hours'] = round(row['minutes'] / 60, 2)
    thr.append(row)
res['throughput'] = thr
res['throughput_conditions'] = {'jev': 'run_screen.mjs with --batch 10 --conc 3 --gap 1900: a global pacing of one request per 1.9 s (at most about 32 requests per minute, about 316 records per minute); the observed rate is lower because of HTTP 429 back-offs (counts in the logs: development 4359, held-out 1152, CLEF 2277 throttled responses). The measured gateway limit was 30 requests per 10 seconds with retry-after (HANDOVER.md section 12).',
                                'chat_models': 'Gateway free tier: about five requests per minute per chat model (HANDOVER.md section 14; 503 responses without retry-after, section 13), one record per request; GPT-4o-mini additionally restricted to the OpenAI provider for log-probabilities.',
                                'claude_cli': 'Subscription command-line interface (claude -p), one record per call, concurrency as set by --conc (default 2 in run_screen.mjs); no gateway limit.',
                                'exclusions': 'Times and costs cover model scoring only; they exclude CPU time for the free rankers and ASReview simulations, retrieval of records, and human screening time.'}
res['manuscript_577_tokens_claim'] = {'origin': 'HANDOVER.md section 11 (gene-trait experiment, same CLI settings): "문맥 577토큰"; not measured in the screening runs (run_screen.mjs stores total_cost_usd and modelId only).', 'recommendation': 'State as a fixed context of about 577 tokens measured in an earlier run with the same interface settings, or remove.'}
json.dump(res, open(OUT + '/results.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
pd.DataFrame(thr).to_csv(OUT + '/throughput.csv', index=False)
print(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
