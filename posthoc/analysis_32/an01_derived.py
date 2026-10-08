# AN-0001-01 (npj round 1): summary quantities derived from stored outputs for the text changes.
# (a) pooled failure rate of tau=0.07 over the two evaluation collections; size of each failure
# (b) cross-tabulation of title-and-abstract positive records by side of tau and final inclusion
# (c) cost relations (full-run costs, records not read per difference in proportion read, cost factors with two denominators)
# (d) source and resolution of the Jev scores (script line, raw probe response, decimals in all stored Jev outputs)
# (e) intervals behind every reliability figure in the main text and Table 2
# Post hoc; no model call. Pilot review: aggregated counts from stored Lancet outputs only.
import json, datetime, re, os
from decimal import Decimal
import numpy as np, pandas as pd
from scipy.stats import beta
ROOT = '/N/project/AiLab/jev'; SYN = ROOT + '/synergy/'; AN = ROOT + '/review_pipeline/rounds/round_0001/revision/analysis/'
TAU = 0.07; Z = 1.959963984540054
def wilson(k, n, z=Z):
    ph = k / n; d = 1 + z * z / n; c = ph + z * z / (2 * n); h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)); return [float((c - h) / d), float((c + h) / d)]
def cp2(k, n):
    return [0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))]
def cp1_lower(k, n):  # one-sided exact lower 95% limit
    return 0.0 if k == 0 else float(beta.ppf(0.05, k, n - k + 1))
def cp1_upper(k, n):  # one-sided exact upper 95% limit
    return 1.0 if k == n else float(beta.ppf(0.95, k + 1, n - k))
def iv(k, n):
    return {'k': k, 'n': n, 'proportion': k / n, 'wilson95_two_sided': wilson(k, n), 'clopper_pearson95_two_sided': cp2(k, n), 'exact_one_sided_lower95': cp1_lower(k, n), 'exact_one_sided_upper95': cp1_upper(k, n)}
out = {'analysis_id': 'AN-0001-01'}
# ---------------- load records and scores ----------------
def load(recf, jevf):
    R = json.load(open(recf)); J = {d['id']: d['p'] for d in json.load(open(jevf)) if d.get('ok') and d.get('p') is not None}
    return R, J
RT, JT = load(SYN + 'test_records.json', SYN + 'jev_test.json'); RC, JC = load(ROOT + '/clef/clef_records.json', SYN + 'jev_clef.json')
def lab(v): return None if v is None or v == '' else int(v)
# ---------------- (a) pooled failures ----------------
def per_review(R, J, key, or0):
    by = {}
    for r in R: by.setdefault(r['review'], []).append(r)
    rows = []
    for k, v in by.items():
        ys = [lab(r[key]) for r in v]
        if any(y is None for y in ys):
            if not or0: continue
            ys = [y or 0 for y in ys]
        y = np.array(ys); p = np.array([J[r['id']] for r in v]); s = p >= TAU
        if y.sum() == 0: continue
        rows.append({'review': k, 'N': len(y), 'n_pos': int(y.sum()), 'found': int(y[s].sum()), 'missed': int(y[~s].sum()), 'read': int(s.sum()), 'recall': float(y[s].sum() / y.sum()), 'work': float(s.mean())})
    return pd.DataFrame(rows)
H = per_review(RT, JT, 'label', False); C = per_review(RC, JC, 'label', True)
fails = pd.concat([H.assign(collection='heldout'), C.assign(collection='clef_content')])
fails = fails[fails.recall < 0.95]
M7 = json.load(open(AN + 'AN-0001-07/results.json'))
study = {}
for cd, v in M7['reviews'].items():
    pr = v.get('published_review', {}); study[cd] = {'n_included_study_ids_represented_in_clef_relevant_set': pr.get('n_included_study_ids_represented_in_clef_relevant_set'), 'n_of_these_with_at_least_one_report_at_or_above_tau': pr.get('n_of_these_with_at_least_one_report_at_or_above_tau'), 'study_level_recall_at_tau': pr.get('study_level_recall_at_tau'), 'n_included_studies_published_review': pr.get('n_included_studies'),
                     'missed_records': [{'pmid': m.get('pmid'), 'p': m.get('jev_probability'), 'category_ai_proposed_unverified': m.get('ai_proposed_category'), 'study_id_in_published_review': m.get('study_id_in_published_review'), 'reference_list_section': m.get('reference_list_section'), 'contribution_to_published_review': m.get('contribution_to_published_review')} for m in v.get('missed', [])]}
nf = len(fails); ntot = len(H) + len(C)
out['a_pooled_failures_final_inclusion'] = {'reviews': ntot, 'heldout_reviews': len(H), 'clef_content_reviews': len(C), 'failures': nf, **iv(nf, ntot),
    'failing_reviews': [{**r, 'published_review_study_level_from_Lancet_AN-0001-07_ai_proposed_unverified': study.get(r['review'])} for r in fails.to_dict('records')],
    'total_positive_records_missed': int(fails.missed.sum()), 'total_positive_records_in_failing_reviews': int(fails.n_pos.sum()),
    'note': 'Final inclusion for SYNERGY+ held-out reviews and the CLEF content-level label for Cochrane reviews. The study-level mapping of missed CLEF records to included studies of the published reviews comes from Lancet AN-0001-07 (AI-assisted, not verified by the authors).'}
# ---------------- (b) cross-tabulation of TA positives ----------------
def xtab(R, J, or0, reviews_filter):
    by = {}
    for r in R: by.setdefault(r['review'], []).append(r)
    t = {'ta_pos_below_final_pos': 0, 'ta_pos_below_final_neg': 0, 'ta_pos_above_final_pos': 0, 'ta_pos_above_final_neg': 0,
         'ta_neg_below_final_pos': 0, 'ta_neg_above_final_pos': 0, 'final_pos_below': 0, 'final_pos_total': 0, 'ta_pos_total': 0, 'n_reviews': 0, 'n_records': 0}
    for k, v in by.items():
        ta = [lab(r['label_ta']) for r in v]
        if any(x is None for x in ta): continue
        fi = [lab(r['label']) for r in v]
        fi = [(x or 0) for x in fi] if or0 else fi
        if any(x is None for x in fi): continue
        if not reviews_filter(ta, fi): continue
        t['n_reviews'] += 1; t['n_records'] += len(v)
        for r, a, f in zip(v, ta, fi):
            below = J[r['id']] < TAU; side = 'below' if below else 'above'
            if a == 1: t[f'ta_pos_{side}_final_{"pos" if f else "neg"}'] += 1; t['ta_pos_total'] += 1
            elif f == 1: t[f'ta_neg_{side}_final_pos'] += 1
            if f == 1: t['final_pos_total'] += 1; t['final_pos_below'] += int(below)
    t['ta_pos_below'] = t['ta_pos_below_final_pos'] + t['ta_pos_below_final_neg']
    return t
out['b_crosstab'] = {'heldout12': xtab(RT, JT, False, lambda ta, fi: sum(ta) > 0),
                     'clef31': xtab(RC, JC, True, lambda ta, fi: sum(ta) > 0)}
R6 = json.load(open(AN + 'AN-0001-06/results.json'))['pilot_review']; b = R6['modes']['batched_10']['0.07']
out['b_crosstab']['pilot_aggregated_batched'] = {'source': 'Lancet AN-0001-06 results.json pilot_review.modes.batched_10.0.07 (aggregated counts only)',
    'ta_pos_total': R6['n_passed_title_abstract'], 'final_pos_total': R6['n_final_included'], 'ta_pos_below': b['passed_ta_below_tau'], 'final_pos_below': b['final_included_below_tau'],
    'ta_pos_below_final_neg': b['fulltext_excluded_below_tau'], 'ta_pos_below_final_pos': b['final_included_below_tau']}
out['b_note'] = ('For the 12 held-out reviews the final label is final inclusion; for the 31 CLEF reviews it is the CLEF content-level label (records without a content-level label counted as 0). '
                 'ta_neg_*_final_pos counts records that are finally positive but not title-and-abstract positive (label inconsistency in the source).')
# ---------------- (c) cost relations ----------------
C12 = json.load(open(AN + 'AN-0001-12/results.json'))
rc = C12['run_costs']; cpk = C12['cost_per_1000']
ST = pd.read_csv(AN + 'AN-0001-08/per_review_stopping.csv')
def records(coll, col):
    T = ST[ST.collection == coll]; return float((T[col] * T.N).sum()) if col.endswith('_work') else float(T[col].sum())
rel = {}
for coll, n_all in (('heldout', 33001), ('clef', None)):
    T = ST[ST.collection == coll]; Nsum = int(T.N.sum())
    thr = float(T.thr_k.sum()); knee = float((T.asr_knee_work * T.N).sum()); stat_jev = float((T.jev_stat_work * T.N).sum()); stat_asr = float((T.asr_stat_work * T.N).sum())
    rel[coll] = {'records': Nsum, 'one_percentage_point_of_records': Nsum / 100,
                 'threshold_records_read': thr, 'knee_asr_records_read_seed_mean': knee, 'records_not_read_threshold_vs_knee_pooled': knee - thr,
                 'macro_pp_threshold_vs_knee': float((T.asr_knee_work - T.thr_work).mean() * 100),
                 'combined_jev_records_read_seed_mean': stat_jev, 'combined_asr_records_read_seed_mean': stat_asr, 'records_not_read_jev_vs_asr_ranking_pooled': stat_asr - stat_jev,
                 'macro_pp_jev_vs_asr_ranking_under_stat': float((T.asr_stat_work - T.jev_stat_work).mean() * 100),
                 'records_not_read_threshold_vs_combined_jev_pooled': stat_jev - thr, 'macro_pp_threshold_vs_combined_jev': float((T.jev_stat_work - T.thr_work).mean() * 100)}
out['c_cost'] = {'full_run_cost_usd': {'heldout_33001_records': rc['jev_held_out']['cost_all_passes_usd'], 'clef_82418_records': rc['jev_clef']['cost_all_passes_usd'], 'development_52957_records': rc['jev_development']['cost_all_passes_usd']},
                 'cost_per_1000_usd': cpk, 'records_relations': rel}
comp = {'gpt4omini_lp': cpk['gpt4omini_lp_single_record'], 'deepseek': cpk['deepseek_single_record'], 'claude_opus_cli_per_2002_attempted': cpk['claude_opus_cli_per_2002_attempted'], 'claude_opus_cli_per_1999_scored': cpk['claude_opus_cli_per_1999_scored']}
out['c_cost']['cost_factors'] = {'denominator_heldout_full_run_0.0201': {k: v / cpk['jev_held_out_batch10'] for k, v in comp.items()},
                                 'denominator_subset_batched_0.0202': {k: v / cpk['jev_p2_batch10_subset'] for k, v in comp.items()},
                                 'denominator_subset_single_record_0.0422': {k: v / cpk['jev_p2_single_record'] for k, v in comp.items()},
                                 'note': 'Ratios of the stored costs per 1000 records (Lancet AN-0001-12 cost_per_1000, rounded there to four decimals); factors computed from the rounded values.'}
# ---------------- (d) rounding ----------------
src = open(SYN + 'run_screen.mjs').read().splitlines()
lines = [{'line': i + 1, 'text': l.strip()} for i, l in enumerate(src) if 'noul' in l]
probe = json.load(open(AN + 'AN-0001-15/probe_record.json'))
pv = [s['response_body']['answers'] for s in probe['steps'] if s.get('step') == 'probe'][0]
files = ['jev_dev2000.json', 'jev_test.json', 'jev_clef.json', 'jev_testcmp_single.json', 'jev_testcmp_b10.json', 'jev_robust_q1.json', 'jev_robust_q2.json', 'jev_robust_titleonly.json', 'jev_robust_mismatch.json', 'dhl_jev_b10.json', 'dhl_jev_b5.json']
dec = {}
def ndec(x):
    s = format(Decimal(repr(float(x))).normalize(), 'f'); return len(s.split('.')[1]) if '.' in s else 0
for f in files + ['../screen/screen_jev.json']:
    path = SYN + f
    if not os.path.exists(path): dec[f] = 'missing'; continue
    vals = [d['p'] for d in json.load(open(path)) if d.get('ok') and d.get('p') is not None and (d.get('backend', 'jev') == 'jev')]
    cnt = {}
    for v in vals: cnt[ndec(v)] = cnt.get(ndec(v), 0) + 1
    dec[f] = {'n_scores': len(vals), 'decimals_count': {str(k): v for k, v in sorted(cnt.items())}, 'distinct_values': len(set(vals)), 'min': float(min(vals)), 'max': float(max(vals)),
              'all_multiples_of_0.01': all(abs(round(v * 100) - v * 100) < 1e-9 for v in vals)}
out['d_rounding'] = {'run_screen_mjs_lines_with_noul': lines, 'probe_response_answers': pv, 'decimals_in_stored_outputs': dec,
                     'reading': 'The script stores answers[r_k].noul as returned (no rounding in the script); the raw probe response of 26 September 2026 returned 0.99. The stored scores are the values returned by the endpoint.'}
# ---------------- (e) intervals behind reliability figures ----------------
figs = {'heldout threshold 23/23': (23, 23), 'heldout knee (ASReview, as adapted) 20/23': (20, 23), 'CLEF threshold 25/28': (25, 28), 'CLEF knee 26/28': (26, 28),
        'heldout title-abstract threshold 7/12': (7, 12), 'CLEF title-abstract threshold 25/31': (25, 31), 'statistical criterion heldout 23/23': (23, 23), 'statistical criterion CLEF 28/28': (28, 28),
        'pooled threshold failures 3/51': (3, 51), 'development threshold 72/73': (72, 73)}
out['e_intervals'] = {k: iv(*v) for k, v in figs.items()}
# ---------------- lookups for E1.23 (stored values, no new computation beyond averaging) ----------------
A1 = json.load(open(AN + 'AN-0001-01/results.json'))['datasets']
out['lookups'] = {'O_E_clef_final_batched_unrounded': A1['clef_final_batched']['O_E_ratio'], 'O_E_heldout_final_batched_unrounded': A1['heldout_final_batched']['O_E_ratio'],
                  'O_E_clef_ta_batched_unrounded': A1['clef_ta_batched']['O_E_ratio'], 'O_E_heldout_ta_batched_unrounded': A1['heldout_ta_batched']['O_E_ratio']}
for split, f in (('heldout', 'asr_test.json'), ('development', 'asr_dev.json')):
    D = pd.DataFrame([d for d in json.load(open(SYN + f)) if d.get('wss95') is not None])
    g = D.groupby(['review', 'method'])['wss95'].mean().unstack()
    out['lookups'][f'{split}_macro_wss95_lambda2'] = float(g['asr_jevblend_2'].mean()) if 'asr_jevblend_2' in g else None
    out['lookups'][f'{split}_macro_wss95_lambda3'] = float(g['asr_jevblend_3'].mean()) if 'asr_jevblend_3' in g else None
    out['lookups'][f'{split}_macro_wss95_lambda1'] = float(g['asr_jevblend_1'].mean()) if 'asr_jevblend_1' in g else None
    out['lookups'][f'{split}_n_reviews_lambda'] = int(g['asr_jevblend_3'].notna().sum())
out['finished'] = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
json.dump(out, open('results.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
