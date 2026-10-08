# AN-0001-09: composition of the LLM comparison subset, pooled and restricted AUCs, refusals, instrument coarseness.
# Reads original outputs read-only from synergy/; writes only into this folder.
import json, os, sys, collections, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
S = '/N/project/AiLab/jev/synergy/'; OUT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(S + 'test_records.json'))
lab = {r['id']: r['label'] for r in R}; rev = {r['id']: r['review'] for r in R}
ids = json.load(open(S + 'test_cmp_ids.json'))
meta = pd.read_csv(S + 'data/metadata/review_metadata.csv').set_index('key')
jev_full_raw = json.load(open(S + 'jev_test.json'))
J = {d['id']: d['p'] for d in jev_full_raw if d.get('ok') and d.get('p') is not None}
batch_of = {d['id']: d.get('batch') for d in jev_full_raw if d.get('ok')}
files = {'claude_opus': 'test_cmp_claude-opus.json', 'gpt4omini_lp': 'test_cmp_gpt4omini_lp.json', 'deepseek': 'test_cmp_deepseek.json',
         'jev_single_p2': 'jev_testcmp_single.json', 'jev_b10_subset_p2': 'jev_testcmp_b10.json'}
raw = {k: json.load(open(S + f)) for k, f in files.items()}
models = {'jev': J}
for k in files: models[k] = {d['id']: d['p'] for d in raw[k] if d.get('ok') and d.get('p') is not None}
res = {'inputs': {'records': 'synergy/test_records.json', 'subset_ids': 'synergy/test_cmp_ids.json', 'jev_full_run': 'synergy/jev_test.json', **{k: 'synergy/' + v for k, v in files.items()}}}
# 1. composition
rows = []
for k in sorted({rev[i] for i in ids}):
    ii = [i for i in ids if rev[i] == k]; n1 = sum(lab[i] for i in ii); full = [r for r in R if r['review'] == k]
    rows.append({'review': k, 'n_records_full': len(full), 'included_full': sum(r['label'] for r in full), 'n_subset': len(ii), 'included_subset': n1, 'excluded_subset': len(ii) - n1,
                 'excluded_sampling_fraction': round((len(ii) - n1) / (len(full) - sum(r['label'] for r in full)), 4)})
comp = pd.DataFrame(rows); comp.to_csv(OUT + '/subset_composition.csv', index=False)
res['subset'] = {'n_ids': len(ids), 'n_included': int(comp.included_subset.sum()), 'n_excluded': int(comp.excluded_subset.sum()), 'reviews': len(comp),
                 'min_excluded_per_review': int(comp.excluded_subset.min()), 'median_excluded_per_review': float(comp.excluded_subset.median()), 'max_excluded_per_review': int(comp.excluded_subset.max()),
                 'reviews_with_lt20_excluded': comp[comp.excluded_subset < 20].review.tolist(), 'reviews_with_lt10_excluded': comp[comp.excluded_subset < 10].review.tolist(),
                 'all_included_present': bool(comp.included_subset.sum() == sum(r['label'] for r in R))}
# 2. refused records (Claude Opus CLI)
ref = [d for d in raw['claude_opus'] if not d.get('ok')]
res['refused_claude'] = []
for d in ref:
    k = d['review']; row = {'review': k, 'label': d['label'], 'review_title': meta.loc[k, 'title'], 'review_type': meta.loc[k, 'review_type'], 'journal': meta.loc[k, 'journal'],
                            'error_prefix': str(d.get('error'))[:90], 'p_jev_full_run': J.get(d['id']), 'p_gpt4omini_lp': models['gpt4omini_lp'].get(d['id']), 'p_deepseek': models['deepseek'].get(d['id']),
                            'p_jev_single_p2': models['jev_single_p2'].get(d['id'])}
    # rank of the record within its review under each model (1 = highest probability), among the full record set of that review
    for m in ['jev']:
        ii = [r['id'] for r in R if r['review'] == k]; ps = np.array([J[i] for i in ii]); row['jev_rank_in_full_review'] = int((ps > J[d['id']]).sum() + 1); row['jev_n_full_review'] = len(ii)
    res['refused_claude'].append(row)
res['refused_claude_handling'] = 'final_eval.py builds the C2 common set as the records scored by all four models, so the three refused records were dropped from every model for C2 (n_common = 1999).'
# 3. GPT-4o-mini missing Yes/No tokens
g = raw['gpt4omini_lp']; pm = [d['pmass'] for d in g if d.get('ok') and 'pmass' in d]
res['gpt4omini_top20'] = {'n_records_in_file': len(g), 'n_ok': sum(1 for d in g if d.get('ok')), 'n_error_in_final_file': sum(1 for d in g if not d.get('ok')),
                          'error_types_in_final_file': collections.Counter(str(d.get('error')) for d in g if not d.get('ok')),
                          'pmass_min': float(min(pm)), 'pmass_median': float(np.median(pm)), 'n_pmass_below_0.99': int(sum(1 for v in pm if v < 0.99)), 'n_pmass_below_0.9': int(sum(1 for v in pm if v < 0.9)),
                          'note': 'The harness (run_screen.mjs askGatewayLP) retries when the top-20 list is missing and records the error "no yes/no in top" when neither token appears; only successful records are kept when a run is resumed. progress.log shows the first gpt-4o-mini run at 19:00 on Sept 24, 2026 with 5 failures among the first 50 attempts (type not recorded); those records were re-scored in the resumed run and the final file has no errors.'}
# 4. common set and macro AUC (replicates final_eval.py C2)
main = ['jev', 'gpt4omini_lp', 'deepseek', 'claude_opus']
common = [i for i in ids if all(i in models[m] for m in main)]
res['n_common'] = len(common)
def macro(idset, ms):
    rr = []
    for k in sorted({rev[i] for i in idset}):
        ii = [i for i in idset if rev[i] == k]; y = np.array([lab[i] for i in ii])
        if 0 < y.sum() < len(y): rr.append({'review': k, 'n': len(ii), 'n1': int(y.sum()), 'n0': int(len(y) - y.sum()), **{m: roc_auc_score(y, [models[m][i] for i in ii]) for m in ms}})
    return pd.DataFrame(rr)
rng = np.random.default_rng(12345)
def boot_diff(d, B=5000):
    d = np.asarray(d); idx = rng.integers(0, len(d), (B, len(d))); m = d[idx].mean(1); return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]
def summarise(B2, ms, tag):
    out = {'n_reviews': len(B2), 'macro_auc': {m: round(float(B2[m].mean()), 4) for m in ms}}
    if 'gpt4omini_lp' in ms: d = B2['jev'] - B2['gpt4omini_lp']; out['jev_minus_gpt4omini'] = {'diff': round(float(d.mean()), 4), 'ci': boot_diff(d), 'wins': int((d > 0).sum())}
    if 'claude_opus' in ms: d = B2['claude_opus'] - B2['jev']; out['claude_minus_jev'] = {'diff': round(float(d.mean()), 4), 'ci': boot_diff(d), 'claude_wins': int((d > 0).sum())}
    if 'deepseek' in ms: d = B2['jev'] - B2['deepseek']; out['jev_minus_deepseek'] = {'diff': round(float(d.mean()), 4), 'ci': boot_diff(d), 'wins': int((d > 0).sum())}
    return out
B_all = macro(common, main + ['jev_single_p2', 'jev_b10_subset_p2']); B_all.to_csv(OUT + '/per_review_auc_common1999.csv', index=False)
res['macro_common'] = summarise(B_all, main, 'common'); res['macro_common']['jev_single_p2_macro'] = round(float(B_all['jev_single_p2'].mean()), 4); res['macro_common']['jev_b10_subset_p2_macro'] = round(float(B_all['jev_b10_subset_p2'].mean()), 4)
B_2002 = macro(ids, ['jev', 'gpt4omini_lp', 'deepseek']); res['macro_all2002_without_claude'] = summarise(B_2002, ['jev', 'gpt4omini_lp', 'deepseek'], 'all')
# 5. restricted macro AUC (reviews with >= 20 excluded records in the subset; also >= 10)
for thr in (20, 10):
    keep = comp[comp.excluded_subset >= thr].review.tolist(); Bk = B_all[B_all.review.isin(keep)]
    res[f'macro_restricted_excluded_ge{thr}'] = {'reviews_kept': keep, **summarise(Bk, main, f'ge{thr}')}
    res[f'macro_restricted_excluded_ge{thr}']['reviews_dropped'] = comp[comp.excluded_subset < thr].review.tolist()
# 6. pooled (micro) AUC on the common set with record-level bootstrap
y = np.array([lab[i] for i in common]); P = {m: np.array([models[m][i] for i in common]) for m in main}
pooled = {m: round(float(roc_auc_score(y, P[m])), 4) for m in main}
rngb = np.random.default_rng(12345); B = 2000; boots = {m: [] for m in main}; n = len(common)
for b in range(B):
    idx = rngb.integers(0, n, n)
    if 0 < y[idx].sum() < n:
        for m in main: boots[m].append(roc_auc_score(y[idx], P[m][idx]))
res['pooled_common'] = {'n': n, 'n_included': int(y.sum()), 'auc': pooled, 'auc_ci_record_bootstrap': {m: [round(float(np.percentile(boots[m], 2.5)), 4), round(float(np.percentile(boots[m], 97.5)), 4)] for m in main},
                        'diff_ci': {'jev_minus_gpt4omini': [round(float(np.percentile(np.array(boots['jev']) - np.array(boots['gpt4omini_lp']), q)), 4) for q in (2.5, 97.5)],
                                    'claude_minus_jev': [round(float(np.percentile(np.array(boots['claude_opus']) - np.array(boots['jev']), q)), 4) for q in (2.5, 97.5)],
                                    'jev_minus_deepseek': [round(float(np.percentile(np.array(boots['jev']) - np.array(boots['deepseek']), q)), 4) for q in (2.5, 97.5)]},
                        'diff_point': {'jev_minus_gpt4omini': round(pooled['jev'] - pooled['gpt4omini_lp'], 4), 'claude_minus_jev': round(pooled['claude_opus'] - pooled['jev'], 4), 'jev_minus_deepseek': round(pooled['jev'] - pooled['deepseek'], 4)},
                        'bootstrap': 'record-level, B=2000, seed 12345, percentile'}
y2 = np.array([lab[i] for i in ids]); res['pooled_all2002_without_claude'] = {m: round(float(roc_auc_score(y2, [models[m][i] for i in ids])), 4) for m in ['jev', 'gpt4omini_lp', 'deepseek']}
# 7. instrument coarseness: distinct values and tied positive-negative pairs within reviews
ties = {}
for m in main + ['jev_single_p2', 'jev_b10_subset_p2']:
    vals = [models[m][i] for i in common]; tied = 0; pairs = 0; per = []
    for k in sorted({rev[i] for i in common}):
        ii = [i for i in common if rev[i] == k]; pos = [models[m][i] for i in ii if lab[i] == 1]; neg = [models[m][i] for i in ii if lab[i] == 0]
        if not pos or not neg: continue
        cp = collections.Counter(pos); cn = collections.Counter(neg); t = sum(cp[v] * cn.get(v, 0) for v in cp); pairs += len(pos) * len(neg); tied += t; per.append(t / (len(pos) * len(neg)))
    ties[m] = {'distinct_values_on_common': len(set(vals)), 'tied_pos_neg_pair_fraction_pooled': round(tied / pairs, 4), 'tied_pair_fraction_mean_over_reviews': round(float(np.mean(per)), 4), 'max_over_reviews': round(float(np.max(per)), 4)}
res['ties'] = ties
res['ties_note'] = 'AUC via sklearn roc_auc_score assigns one half to tied positive-negative pairs (mid-rank convention); DeepSeek and Claude returned verbalised probabilities with few distinct values, GPT-4o-mini log-probability ratios are effectively continuous, Jev returns two-decimal probabilities.'
# 8. scoring mode of the Jev subset AUC
res['jev_subset_scoring_mode'] = {'final_eval_source': 'final_eval.py: models = {"jev": J} where J is loaded from jev_test.json (the full held-out run, ten records per request)',
                                  'subset_records_with_batch10_in_full_run': int(sum(1 for i in ids if batch_of.get(i) == 10)), 'n_subset': len(ids),
                                  'single_record_rescoring_p2': 'jev_testcmp_single.json (batch 1) and jev_testcmp_b10.json (batch 10 within the subset); post_test.json P2 macro AUC single 0.9601, b10_subset 0.9625, b10_full 0.9615 on 23 reviews (2002 records)'}
# per-review distinct values of DeepSeek/Claude on subset for the appendix table
comp2 = comp.merge(B_all[['review', 'jev', 'gpt4omini_lp', 'deepseek', 'claude_opus']], on='review', how='left'); comp2.to_csv(OUT + '/subset_composition_with_auc.csv', index=False)
json.dump(res, open(OUT + '/results.json', 'w'), indent=1, default=lambda o: o if not hasattr(o, 'item') else o.item())
print(json.dumps(res, indent=1, default=lambda o: o if not hasattr(o, 'item') else o.item()))
