# AN-0001-08 (npj round 1): sensitivity of the paired tests to the treatment of zero differences.
# Input: fx01_exact_paired.json written in this folder by fx01_exact_paired_copy.py (copy of the Lancet recount with the
# per-review exact differences stored; counts and P values checked against the stored Lancet CSV), the per-review file of
# AN-0001-02 (new comparisons with the knee method from 1000 records and the reliability indicators of C4), and, when
# present, the per-review file of AN-0001-03 (ten-seed hybrid).
# Tests per comparison: (1) Wilcoxon signed-rank, SciPy defaults (zero_method 'wilcox', zeros discarded; as reported);
# (2) Wilcoxon with Pratt's treatment of zeros (zero_method 'pratt'); (3) sign-flip permutation test of the mean difference
# with zeros retained (exact enumeration when at most 20 non-zero differences, otherwise 100,000 random sign flips,
# seed 20260928, P = (1 + count) / (1 + B)). Two-sided throughout. Post hoc; no model call.
import json, os, sys, warnings, datetime, itertools
from fractions import Fraction as F
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260928; BMC = 100000; MAXEXACT = 20
AN = '/N/project/AiLab/jev/review_pipeline_npj/rounds/round_0001/revision/analysis'
LANCET_CSV = '/N/project/AiLab/jev/review_pipeline/rounds/round_0001/revision/analysis/audit_fix_iter2/fx01_exact_paired.csv'
X = json.load(open(os.path.join(HERE, 'fx01_exact_paired.json')))
A = pd.read_csv(os.path.join(HERE, 'fx01_exact_paired.csv')); L = pd.read_csv(LANCET_CSV)
cols = ['exact_higher', 'exact_lower', 'exact_tied', 'exact_p']
repro = bool((A.key.values == L.key.values).all() and A[cols].fillna(-1).equals(L[cols].fillna(-1)))
def auto_method(d):
    nz = d[d != 0]; ties = len(nz) - len(np.unique(np.abs(nz))); zeros = int((d == 0).sum())
    if zeros == 0 and ties == 0 and len(d) <= 50: return 'exact'
    if len(d) <= 13: return 'exact (permutation; zero or tied differences)'
    return 'asymptotic'
def tests(d):
    d = np.asarray(d, float); n = len(d); nz = d[d != 0]; m = len(nz); r = {'n': n, 'n_nonzero': m, 'n_zero': n - m, 'mean_diff': float(d.mean()) if n else None}
    if m == 0: r.update({'p_wilcoxon_default': None, 'p_pratt': None, 'p_signflip': 1.0, 'signflip_method': 'all differences zero'}); return r
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r['p_wilcoxon_default'] = float(wilcoxon(d).pvalue); r['wilcoxon_default_method'] = auto_method(d)
        try:
            r['p_pratt'] = float(wilcoxon(d, zero_method='pratt').pvalue)
        except Exception as e:
            r['p_pratt'] = None; r['pratt_error'] = str(e)[:120]
        try:
            r['p_pratt_asymptotic'] = float(wilcoxon(d, zero_method='pratt', method='asymptotic').pvalue)
        except Exception as e:
            r['p_pratt_asymptotic'] = None
        if r.get('p_pratt') is not None and r.get('p_pratt_asymptotic') is not None:
            r['pratt_auto_equals_asymptotic'] = abs(r['p_pratt'] - r['p_pratt_asymptotic']) <= 1e-15
    s_obs = abs(nz.sum()); tol = 1e-12 * max(1.0, np.abs(nz).sum())
    if m <= MAXEXACT:
        signs = 1.0 - 2.0 * ((np.arange(2 ** m, dtype=np.int64)[:, None] >> np.arange(m, dtype=np.int64)) & 1)
        S = np.abs(signs @ nz); r['p_signflip'] = float(np.mean(S >= s_obs - tol)); r['signflip_method'] = f'exact enumeration of 2^{m} sign patterns'
    else:
        rng = np.random.default_rng(SEED); signs = rng.choice([1.0, -1.0], size=(BMC, m))
        S = np.abs(signs @ nz); c = int(np.sum(S >= s_obs - tol)); r['p_signflip'] = float((1 + c) / (1 + BMC)); r['signflip_method'] = f'Monte Carlo, {BMC} random sign flips, seed {SEED}'
    return r
# main-text and Table 2 comparisons, and the confirmatory label
MAIN = {'C1 AUC Jev vs bge-base, 23 held-out reviews': ('Table 2 C1; Results', 'confirmatory (C1: Wilcoxon P<0.01 is part of the criterion)'),
        'C2 AUC Jev vs GPT-4o-mini (log-probability), 23 reviews, 1999 common records (descriptive, no test in v0)': ('Table 2 C2', 'exploratory (C2 has no test in the plan)'),
        'C2 AUC Claude Opus vs Jev, 23 reviews (descriptive, no test in v0)': ('Table 2 C2', 'exploratory (C2 has no test in the plan)'),
        'C3 WSS@95 hybrid vs ASReview default, 23 held-out reviews': ('Table 2 C3; Results', 'confirmatory (C3: Wilcoxon P<0.05 for WSS@95 is part of the criterion)'),
        'C3 WSS@100 hybrid vs ASReview default, 23 held-out reviews (no test in v0)': ('Table 2 C3', 'exploratory (the WSS@100 part of C3 has no test in the plan)'),
        'heldout|paired_threshold_vs_knee_asr|work': ('Table 2 C4 records read', 'exploratory (C4 has no test in the plan)'),
        'heldout_final|jevonly_minus_asr|wss95': ('Results (post hoc Jev ranking vs ASReview)', 'exploratory'),
        'heldout_final|hybrid_minus_jevonly|wss95': ('Results (hybrid increment over the Jev ranking, P=0.025)', 'exploratory'),
        'CLEF AUC Jev vs bge-base, final inclusion, 28 reviews': ('Results (Cochrane)', 'exploratory'),
        'CLEF wss95 asr_jevblend_3 vs asr_prior, 28 reviews': ('Results (Cochrane)', 'exploratory'),
        'clef_final|jevonly_minus_asr|wss95': ('Results (Cochrane)', 'exploratory'),
        'clef_final|hybrid_minus_jevonly|wss95': ('Discussion (0.019 of 0.044 in Cochrane reviews)', 'exploratory'),
        'heldout|stat_jev_only_minus_stat_asreview_work': ('Discussion (2.5 percentage points)', 'exploratory'),
        'clef|stat_jev_only_minus_stat_asreview_work': ('Discussion (0.9 percentage points)', 'exploratory'),
        'heldout|stat_jev_only_minus_threshold_work': ('Discussion (28.8 percentage points)', 'exploratory'),
        'clef|stat_jev_only_minus_threshold_work': ('Discussion (47.0 percentage points)', 'exploratory'),
        'heldout|atd_jev_minus_asr': ('Discussion (average time to discovery)', 'exploratory')}
rows = []
for c in X['comparisons']:
    if c.get('per_review_diff_float') is None: continue
    d = np.array(c['per_review_diff_float'], float)
    if c.get('p_exact_arithmetic') is None and len(d) <= 5: continue
    t = tests(d)
    if c.get('p_exact_arithmetic') is not None and t['p_wilcoxon_default'] is not None and abs(c['p_exact_arithmetic'] - t['p_wilcoxon_default']) > 1e-12 * max(1, c['p_exact_arithmetic']):
        raise SystemExit(f"STOP: default Wilcoxon P differs from the recount: {c['key']}")
    loc, lab = MAIN.get(c['key'], ('', 'exploratory'))
    rows.append({'source': c['analysis'], 'key': c['key'], 'main_text_or_table2': loc, 'label': lab, 'higher': c['higher'], 'lower': c['lower'], 'tied': c['tied'], **t})
# new comparisons from AN-0001-02 (per-review values are seed means of integer ratios; equal values give equal floats)
P2 = pd.read_csv(os.path.join(AN, 'AN-0001-02', 'per_review.csv'))
for coll in ('heldout', 'clef'):
    T = P2[P2.collection == coll]
    for key, a, b, loc, lab in [('threshold_minus_knee1000_asr|work', 'thr_work', 'asr_knee1000_work', 'SI Note 11 (AN-0001-02)', 'exploratory'),
                                 ('knee1000_minus_knee150_asr|work', 'asr_knee1000_work', 'asr_knee150_work', 'SI Note 11 (AN-0001-02)', 'exploratory')]:
        d = (T[a] - T[b]).values; t = tests(d)
        rows.append({'source': 'AN-0001-02 (npj)', 'key': f'{coll}|{key}', 'main_text_or_table2': loc, 'label': lab, 'higher': int((d > 0).sum()), 'lower': int((d < 0).sum()), 'tied': int((d == 0).sum()), **t})
    for key, a, b, loc, lab in [('reliability|threshold_minus_knee150_asr', 'thr_rec', 'asr_knee150_rec', 'Table 2 C4 reliability' if coll == 'heldout' else 'Results (Cochrane)', 'exploratory (C4 has no test in the plan)' if coll == 'heldout' else 'exploratory'),
                                 ('reliability|threshold_minus_knee1000_asr', 'thr_rec', 'asr_knee1000_rec', 'SI Note 11 (AN-0001-02)', 'exploratory')]:
        ia = (T[a] >= 0.95 - 1e-12).astype(int).values; ib = (T[b] >= 0.95 - 1e-12).astype(int).values; d = (ia - ib).astype(float); t = tests(d)
        rows.append({'source': 'AN-0001-02 (npj)', 'key': f'{coll}|{key}', 'main_text_or_table2': loc, 'label': lab, 'higher': int((d > 0).sum()), 'lower': int((d < 0).sum()), 'tied': int((d == 0).sum()), **t})
# ten-seed hybrid comparisons from AN-0001-03 (bundle H), when present
p3 = os.path.join(AN, 'AN-0001-03', 'per_review.csv'); an03_used = False; an03_cols = None
if os.path.exists(p3) and os.path.exists(os.path.join(AN, 'AN-0001-03', 'results.json')):
    T3 = pd.read_csv(p3); an03_cols = list(T3.columns)
    pairs = []
    for coll in sorted(T3['collection'].unique()) if 'collection' in T3 else []:
        S = T3[T3.collection == coll]
        for met in ('wss95', 'wss100'):
            for a, b in [(f'hybrid10_{met}', f'asr10_{met}'), (f'hybrid10_{met}', f'jevonly10_{met}')]:   # column names of npj AN-0001-03 per_review.csv
                if a in S and b in S:
                    d = (S[a] - S[b]).values; t = tests(d); an03_used = True
                    rows.append({'source': 'AN-0001-03 (npj)', 'key': f'{coll}|{a}_minus_{b}', 'main_text_or_table2': 'SI Note 10 (AN-0001-03)', 'label': 'exploratory', 'higher': int((d > 0).sum()), 'lower': int((d < 0).sum()), 'tied': int((d == 0).sum()), **t})
R = pd.DataFrame(rows)
R['sig05_default'] = R.p_wilcoxon_default < 0.05; R['sig05_pratt'] = R.p_pratt < 0.05; R['sig05_signflip'] = R.p_signflip < 0.05
R['conclusion_at_0.05_changes'] = (R.sig05_default != R.sig05_pratt) | (R.sig05_default != R.sig05_signflip)
R.to_csv(os.path.join(HERE, 'zero_sensitivity_all.csv'), index=False)
M = R[R.main_text_or_table2 != '']
MT = M[~M.main_text_or_table2.str.startswith('SI')]   # main text and Table 2 only (SI rows of AN-0001-02 excluded)
M.to_csv(os.path.join(HERE, 'zero_sensitivity_main.csv'), index=False)
out = {'analysis_id': 'AN-0001-08', 'fx01_copy_reproduces_lancet_counts_and_p': repro, 'n_comparisons_tested': int(len(R)), 'n_main_text_or_table2': int(len(M)),
       'n_with_zero_differences': int((R.n_zero > 0).sum()), 'n_conclusion_at_0.05_changes_all': int(R['conclusion_at_0.05_changes'].sum()),
       'n_conclusion_at_0.05_changes_main_and_si_rows': int(M['conclusion_at_0.05_changes'].sum()), 'n_main_text_or_table2_only': int(len(MT)),
       'n_conclusion_at_0.05_changes_main_text_or_table2_only': int(MT['conclusion_at_0.05_changes'].sum()), 'changes_main_text_or_table2_only': MT[MT['conclusion_at_0.05_changes']][['key', 'n', 'n_zero', 'p_wilcoxon_default', 'p_pratt', 'p_signflip']].to_dict('records'),
       'n_with_pratt_differing_from_default_at_0.05': int((R.sig05_default != R.sig05_pratt).sum()), 'n_with_signflip_differing_from_default_at_0.05': int((R.sig05_default != R.sig05_signflip).sum()),
       'n_changes_in_comparisons_without_zeros': int((R['conclusion_at_0.05_changes'] & (R.n_zero == 0)).sum()),
       'changes_all': R[R['conclusion_at_0.05_changes']][['key', 'n', 'n_zero', 'p_wilcoxon_default', 'p_pratt', 'p_signflip']].to_dict('records'),
       'main': M.to_dict('records'), 'an03_used': an03_used, 'an03_columns_seen': an03_cols,
       'settings': {'signflip_exact_max_nonzero': MAXEXACT, 'signflip_mc_B': BMC, 'seed': SEED, 'statistic': 'absolute sum of differences (equivalent to the mean, zeros retained)'},
       'confirmatory_definition': 'Confirmatory = a test named in a decision criterion of the plan: C1 (Wilcoxon P<0.01) and the WSS@95 part of C3 (P<0.05). All other P values are exploratory. The composite decision rule with fixed thresholds and disjunctions does not control the overall type I error rate.',
       'finished': datetime.datetime.now().astimezone().isoformat(timespec='seconds')}
json.dump(out, open(os.path.join(HERE, 'results.json'), 'w'), indent=1, default=float)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
print('reproduces Lancet fx01:', repro, '| tested', len(R), '| with zeros', out['n_with_zero_differences'], '| conclusion changes all/main', out['n_conclusion_at_0.05_changes_all'], out['n_conclusion_at_0.05_changes_main_text_or_table2_only'], '| pratt/signflip differ', out['n_with_pratt_differing_from_default_at_0.05'], out['n_with_signflip_differing_from_default_at_0.05'], '| changes without zeros', out['n_changes_in_comparisons_without_zeros'], '| AN-03 used', an03_used)
print(M[['key', 'label', 'n', 'n_zero', 'p_wilcoxon_default', 'p_pratt', 'p_signflip', 'signflip_method']].to_string())
print(pd.DataFrame(out['changes_all']).to_string())
