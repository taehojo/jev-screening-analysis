#!/usr/bin/env python
# AN-0001-13: document the statistical procedures from the original code, recompute exact p-values from stored
# per-review values, reproduce the stored bootstrap CIs by replaying the original RNG call sequences, add the missing
# CIs (C3 WSS@100, prespecified-comparator WSS@100) and the C2 per-review counts, and check AUC tie handling.
# Reads original outputs only (read-only); writes into this folder only.
import os, sys, json, math, time, platform, datetime, re, warnings
import numpy as np, pandas as pd, scipy, sklearn
from scipy.stats import wilcoxon, rankdata
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = '/N/project/AiLab/jev'; SYN = ROOT + '/synergy'
B = 5000; FRESH_SEED = 1013
t0 = time.time(); started = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
def log(*a): print(*a, flush=True)
tests = []; cis = []

def boot_rng(d, rng, Bn=B):
    # identical to boot_diff() in final_eval.py and ci() in clef_eval.py / post_eval.py (before their rounding)
    d = np.asarray(d, float); idx = rng.integers(0, len(d), (Bn, len(d))); m = d[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

def fmt_p(p):
    if p is None: return None
    return '<0.0001' if p < 0.0001 else f'{p:.2g}'

def wilcox(name, a, b, stored_p=None, where=None):
    a = np.asarray(a, float); b = np.asarray(b, float); d = a - b; n = len(d)
    zeros = int((d == 0).sum()); dnz = np.abs(d[d != 0]); ties = int(len(dnz) - len(np.unique(dnz)))
    row = {'test': name, 'n': n, 'zeros': zeros, 'ties_among_nonzero_abs_diffs': ties, 'wins_a_gt_b': int((d > 0).sum()), 'losses_a_lt_b': int((d < 0).sum()), 'mean_diff': float(d.mean()), 'stored_p': stored_p, 'stored_where': where}
    if (d != 0).sum() == 0: row.update({'p_auto': None, 'method_auto': 'not applicable'}); tests.append(row); return row
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = wilcoxon(a, b); row['statistic'] = float(r.statistic); row['p_auto'] = float(r.pvalue)
        row['method_auto_inferred'] = 'exact' if (n <= 50 and zeros == 0 and ties == 0) else ('exact_permutation_ties_or_zeros' if n <= 13 else 'asymptotic')   # SciPy 1.18.1 auto rule from the docstring Notes
        try: row['p_exact'] = float(wilcoxon(a, b, method='exact').pvalue)
        except Exception as e: row['p_exact'] = None; row['p_exact_error'] = repr(e)
        row['p_asymptotic'] = float(wilcoxon(a, b, method='asymptotic').pvalue)
    row['p_auto_equals_exact'] = (row['p_exact'] is not None and abs(row['p_auto'] - row['p_exact']) < 1e-15)
    row['p_auto_equals_asymptotic'] = abs(row['p_auto'] - row['p_asymptotic']) < 1e-15
    row['stored_matches_auto'] = (stored_p is not None and abs(stored_p - row['p_auto']) <= 1e-12 * max(1.0, abs(stored_p)))
    row['p_formatted'] = fmt_p(row['p_auto']); tests.append(row); return row

def ci_check(name, computed, stored, where):
    c = [round(x, 4) for x in computed]; ok = (stored is not None and all(abs(c[i] - stored[i]) < 1e-9 for i in range(2)))
    cis.append({'ci': name, 'computed_4dp': c, 'stored': stored, 'stored_where': where, 'reproduced': bool(ok)}); return c

def load_by(path):
    by = {}
    for r in json.load(open(path)): by.setdefault(r['review'], []).append(r)
    return by
def load_jev(path): return {d['id']: d['p'] for d in json.load(open(path)) if d.get('ok') and d.get('p') is not None}

out = {'analysis_id': 'AN-0001-13', 'started': started, 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'executable': sys.executable}}
F = json.load(open(SYN + '/final_test.json')); CL = json.load(open(SYN + '/clef_results.json')); PT = json.load(open(SYN + '/post_test.json')); PC = json.load(open(SYN + '/prereg_comparator_test.json'))
RB = json.load(open(SYN + '/robust_results.json')); FD = json.load(open(SYN + '/final_dev.json'))

# ---------- 0. SciPy wilcoxon documentation of the 'auto' rule (installed version)
doc = wilcoxon.__doc__; notes = doc[doc.find('Notes'):]
auto_par = [p.strip() for p in re.split(r'\n\s*\n', notes) if '"auto"' in p or "'auto'" in p or 'auto' in p][:3]
out['scipy_wilcoxon_auto_rule_from_docstring'] = auto_par
import inspect; out['scipy_wilcoxon_signature'] = str(inspect.signature(wilcoxon))

# ---------- 1. C1 replay (final_eval.py: rng = default_rng(12345); boot_diff call order C1, C2 gpt, C2 claude, C2 deepseek, C3 wss95)
by = load_by(SYN + '/test_records.json'); J = load_jev(SYN + '/jev_test.json')
revs = [k for k, v in by.items() if all(r['id'] in J for r in v) and 0 < sum(int(r['label']) for r in v) < len(v)]
rows = []
for k in revs:
    v = by[k]; y = np.array([int(r['label']) for r in v]); S = json.load(open(f'{SYN}/scores_zs/test_{k}.json'))
    rows.append({'review': k, 'N': len(v), 'n1': int(y.sum()), 'jev': roc_auc_score(y, [J[r['id']] for r in v]), **{m: roc_auc_score(y, s) for m, s in S.items()}})
A = pd.DataFrame(rows); rng = np.random.default_rng(12345)
d1 = A['jev'] - A['bge']; c = ci_check('C1 AUC Jev minus bge (held-out)', boot_rng(d1, rng), F['C1']['ci'], 'final_test.json C1.ci')
w1 = wilcox('C1 AUC Jev vs bge-base, 23 held-out reviews', A['jev'], A['bge'], F['C1']['wilcoxon_p'], 'final_test.json C1.wilcoxon_p')
out['C1'] = {'macro_jev': float(A.jev.mean()), 'macro_bge': float(A.bge.mean()), 'diff': float(d1.mean()), 'ci_replay': c, 'wins': int((d1 > 0).sum()), 'p': w1['p_auto'], 'p_formatted': w1['p_formatted'], 'method': w1['method_auto_inferred']}
A.to_csv(HERE + '/c1_per_review_auc.csv', index=False)
# ---------- 2. C2 replay and per-review counts
R = json.load(open(SYN + '/test_records.json')); ids = set(json.load(open(SYN + '/test_cmp_ids.json'))); lab = {r['id']: int(r['label']) for r in R}; rev = {r['id']: r['review'] for r in R}
models = {'jev': J}
for name, f in [('gpt4omini_lp', 'test_cmp_gpt4omini_lp.json'), ('deepseek', 'test_cmp_deepseek.json'), ('claude_opus', 'test_cmp_claude-opus.json')]:
    models[name] = {d['id']: d['p'] for d in json.load(open(f'{SYN}/{f}')) if d.get('ok') and d.get('p') is not None}
common = [i for i in ids if all(i in M for M in models.values())]
rr = []
for k in sorted({rev[i] for i in common}):
    ii = [i for i in common if rev[i] == k]; y = np.array([lab[i] for i in ii])
    if 0 < y.sum() < len(y): rr.append({'review': k, 'n_records': len(ii), 'n_included': int(y.sum()), **{m: roc_auc_score(y, [M[i] for i in ii]) for m, M in models.items()}})
B2 = pd.DataFrame(rr)
dg = B2['jev'] - B2['gpt4omini_lp']; cg = ci_check('C2 AUC Jev minus GPT-4o-mini', boot_rng(dg, rng), F['C2_vs_gpt4omini']['ci'], 'final_test.json C2_vs_gpt4omini.ci')
dc = B2['claude_opus'] - B2['jev']; cc = ci_check('C2 AUC Claude Opus minus Jev', boot_rng(dc, rng), F['C2_vs_claude']['ci'], 'final_test.json C2_vs_claude.ci')
dd = B2['jev'] - B2['deepseek']; cd = ci_check('C2 AUC Jev minus DeepSeek', boot_rng(dd, rng), F['C2_vs_deepseek']['ci'], 'final_test.json C2_vs_deepseek.ci')
wg = wilcox('C2 AUC Jev vs GPT-4o-mini (log-probability), 23 reviews, 1999 common records (descriptive, no test in v0)', B2['jev'], B2['gpt4omini_lp'])
wc = wilcox('C2 AUC Claude Opus vs Jev, 23 reviews (descriptive, no test in v0)', B2['claude_opus'], B2['jev'])
wd = wilcox('C2 AUC Jev vs DeepSeek, 23 reviews (descriptive, no test in v0)', B2['jev'], B2['deepseek'])
out['C2'] = {'n_common_records': len(common), 'n_reviews': int(len(B2)), 'macro_auc': {m: float(B2[m].mean()) for m in models},
             'jev_minus_gpt4omini': {'diff': float(dg.mean()), 'ci_replay': cg, 'wins_jev': int((dg > 0).sum()), 'losses': int((dg < 0).sum()), 'ties': int((dg == 0).sum()), 'p_descriptive': wg['p_auto'], 'p_formatted': wg['p_formatted'], 'method': wg['method_auto_inferred']},
             'claude_minus_jev': {'diff': float(dc.mean()), 'ci_replay': cc, 'wins_claude': int((dc > 0).sum()), 'losses': int((dc < 0).sum()), 'ties': int((dc == 0).sum()), 'p_descriptive': wc['p_auto'], 'p_formatted': wc['p_formatted'], 'method': wc['method_auto_inferred']},
             'jev_minus_deepseek': {'diff': float(dd.mean()), 'ci_replay': cd, 'wins_jev': int((dd > 0).sum()), 'losses': int((dd < 0).sum()), 'ties': int((dd == 0).sum()), 'p_descriptive': wd['p_auto'], 'p_formatted': wd['p_formatted'], 'method': wd['method_auto_inferred']}}
B2.to_csv(HERE + '/c2_per_review_auc.csv', index=False)
# ---------- 3. C3 replay and the missing WSS@100 CI
D = pd.DataFrame([d for d in json.load(open(SYN + '/asr_test.json')) if d.get('wss95') is not None and d['review'] in revs])
g = D.groupby(['review', 'method'])[['wss95', 'wss100', 'rec@10', 'rec@20', 'rec@30', 'knee_work', 'knee_rec']].mean().unstack('method')
x = g[('wss95', 'asr_jevblend_3')]; b = g[('wss95', 'asr_prior')]; ok = x.notna() & b.notna(); d95 = (x - b)[ok]
c95 = ci_check('C3 WSS@95 hybrid minus ASReview', boot_rng(d95, rng), F['C3']['ci'], 'final_test.json C3.ci')
d100 = (g[('wss100', 'asr_jevblend_3')] - g[('wss100', 'asr_prior')])[ok]
c100_seq = boot_rng(d100, rng)                                   # next call in the same rng sequence (seed 12345)
c100_fresh = boot_rng(d100, np.random.default_rng(FRESH_SEED))   # independent seed
w95 = wilcox('C3 WSS@95 hybrid vs ASReview default, 23 held-out reviews', x[ok], b[ok], F['C3']['wilcoxon_p'], 'final_test.json C3.wilcoxon_p')
w100 = wilcox('C3 WSS@100 hybrid vs ASReview default, 23 held-out reviews (no test in v0)', g[('wss100', 'asr_jevblend_3')][ok], g[('wss100', 'asr_prior')][ok])
out['C3'] = {'n': int(ok.sum()), 'wss95_diff': float(d95.mean()), 'wss95_ci_replay': c95, 'wss95_wins': int((d95 > 0).sum()), 'wss95_p': w95['p_auto'], 'wss95_p_formatted': w95['p_formatted'], 'wss95_method': w95['method_auto_inferred'],
             'wss100_hybrid': float(g[('wss100', 'asr_jevblend_3')][ok].mean()), 'wss100_asr': float(g[('wss100', 'asr_prior')][ok].mean()), 'wss100_diff': float(d100.mean()),
             'wss100_ci_seed12345_next_call': [round(v, 4) for v in c100_seq], 'wss100_ci_seed1013': [round(v, 4) for v in c100_fresh],
             'wss100_wins': int((d100 > 0).sum()), 'wss100_losses': int((d100 < 0).sum()), 'wss100_ties': int((d100 == 0).sum()), 'wss100_p': w100['p_auto'], 'wss100_p_formatted': w100['p_formatted'], 'wss100_method': w100['method_auto_inferred']}
# ---------- 4. prespecified comparators (al_test_prereg.json): CI for WSS@100 differences; seed of the stored CI not recorded
D2 = pd.DataFrame([d for d in json.load(open(SYN + '/al_test_prereg.json')) if d.get('wss95') is not None])
g2 = D2.groupby(['review', 'method'])[['wss95', 'wss100', 'knee_work', 'knee_rec']].mean().unstack('method')
out['prereg_comparators'] = {}
for m, tag in [('al|lr|oracle|0|0', 'logistic_regression'), ('al|nb|oracle|0|0', 'naive_bayes')]:
    cw = g2[('wss95', m)].reindex(x.index); c100 = g2[('wss100', m)].reindex(x.index)
    e95 = (x - cw); e100 = (g[('wss100', 'asr_jevblend_3')] - c100)
    wa = wilcox(f'Hybrid vs prespecified comparator {tag}, WSS@95, 23 held-out reviews', x, cw, PC[m]['p'], f'prereg_comparator_test.json {m}.p')
    wb = wilcox(f'Hybrid vs prespecified comparator {tag}, WSS@100, 23 held-out reviews (no test in v0)', g[('wss100', 'asr_jevblend_3')], c100)
    kn = g2[('knee_rec', m)].reindex(x.index)
    out['prereg_comparators'][tag] = {'comparator_wss95': float(cw.mean()), 'comparator_wss100': float(c100.mean()), 'wss95_diff': float(e95.mean()), 'wss95_ci_seed1013': [round(v, 4) for v in boot_rng(e95, np.random.default_rng(FRESH_SEED))],
        'stored_wss95_ci_seed_unknown': PC[m]['ci'], 'wss95_wins': int((e95 > 0).sum()), 'wss95_p': wa['p_auto'], 'wss95_p_formatted': wa['p_formatted'], 'wss95_method': wa['method_auto_inferred'], 'stored_p': PC[m]['p'],
        'wss100_diff': float(e100.mean()), 'wss100_ci_seed1013': [round(v, 4) for v in boot_rng(e100, np.random.default_rng(FRESH_SEED))], 'wss100_wins': int((e100 > 0).sum()), 'wss100_losses': int((e100 < 0).sum()), 'wss100_p': wb['p_auto'], 'wss100_p_formatted': wb['p_formatted'], 'wss100_method': wb['method_auto_inferred'],
        'knee_reliability_count': int((kn >= 0.95).sum()), 'knee_reliability': float((kn >= 0.95).mean()), 'knee_work': float(g2[('knee_work', m)].reindex(x.index).mean()), 'stored': PC[m]}
# ---------- 5. CLEF replay (clef_eval.py: rng = default_rng(11); ci() order: label, label_ta, then AL wss95/wss100 vs prior, wss95/wss100 vs pseudo)
rngc = np.random.default_rng(11)
Tl = pd.read_csv(SYN + '/clef_zs_label.csv'); Tt = pd.read_csv(SYN + '/clef_zs_label_ta.csv')
dl = Tl.jev_AUC - Tl.bge_AUC; ci_check('CLEF AUC Jev minus bge, final inclusion (28)', boot_rng(dl, rngc), CL['zeroshot_label']['jev_vs_bge_auc']['ci'], 'clef_results.json zeroshot_label.jev_vs_bge_auc.ci')
dt = Tt.jev_AUC - Tt.bge_AUC; ci_check('CLEF AUC Jev minus bge, title-abstract labels (31)', boot_rng(dt, rngc), CL['zeroshot_label_ta']['jev_vs_bge_auc']['ci'], 'clef_results.json zeroshot_label_ta.jev_vs_bge_auc.ci')
wilcox('CLEF AUC Jev vs bge-base, final inclusion, 28 reviews', Tl.jev_AUC, Tl.bge_AUC, CL['zeroshot_label']['jev_vs_bge_auc']['p'], 'clef_results.json zeroshot_label.jev_vs_bge_auc.p')
wilcox('CLEF AUC Jev vs bge-base, title-abstract labels, 31 reviews', Tt.jev_AUC, Tt.bge_AUC, CL['zeroshot_label_ta']['jev_vs_bge_auc']['p'], 'clef_results.json zeroshot_label_ta.jev_vs_bge_auc.p')
Ac = pd.DataFrame([d for d in json.load(open(SYN + '/asr_clef.json')) if d.get('wss95') is not None])
gc = Ac.groupby(['review', 'method'])[['wss95', 'wss100', 'knee_work', 'knee_rec']].mean().unstack('method')
out['CLEF'] = {}
for a_, b_ in [('asr_jevblend_3', 'asr_prior'), ('asr_jevblend_3', 'asr_pseudo_10_50')]:
    for met in ['wss95', 'wss100']:
        xx, zz = gc[(met, a_)], gc[(met, b_)]; okc = xx.notna() & zz.notna(); dcl = (xx - zz)[okc]; key = f'{met}_{a_}_vs_{b_}'
        cc_ = ci_check(f'CLEF {key}', boot_rng(dcl, rngc), CL[key]['ci'], f'clef_results.json {key}.ci')
        w = wilcox(f'CLEF {met} {a_} vs {b_}, 28 reviews', xx[okc], zz[okc], CL[key]['p'], f'clef_results.json {key}.p')
        out['CLEF'][key] = {'n': int(okc.sum()), 'diff': float(dcl.mean()), 'ci_replay': cc_, 'wins': int((dcl > 0).sum()), 'losses': int((dcl < 0).sum()), 'p': w['p_auto'], 'p_formatted': w['p_formatted'], 'method': w['method_auto_inferred']}
# ---------- 6. post_eval replay (rng = default_rng(7); paired() order P1 wss95 x2, P1 wss100 x2, P4 wss95 x2, P4 wss100 x1, P3 x2, P2 x2)
rngp = np.random.default_rng(7)
def rep(name, a, b, stored, where):
    okp = a.notna() & b.notna(); dp = (a - b)[okp]; c_ = ci_check(name, boot_rng(dp, rngp), stored['ci'], where + '.ci')
    w = wilcox(name, a[okp], b[okp], stored.get('wilcoxon_p'), where + '.wilcoxon_p')
    return {'n': int(okp.sum()), 'diff': float(dp.mean()), 'ci_replay': c_, 'wins': int((dp > 0).sum()), 'losses': int((dp < 0).sum()), 'p': w['p_auto'], 'p_formatted': w['p_formatted'], 'method': w['method_auto_inferred']}
out['post_hoc'] = {}
out['post_hoc']['P1_wss95_hybrid_vs_pseudo'] = rep('P1 WSS@95 hybrid vs weak supervision, 23', g[('wss95', 'asr_jevblend_3')], g[('wss95', 'asr_pseudo_10_50')], PT['P1_wss95'][0], 'post_test.json P1_wss95[0]')
out['post_hoc']['P1_wss95_pseudo_vs_asr'] = rep('P1 WSS@95 weak supervision vs ASReview, 23', g[('wss95', 'asr_pseudo_10_50')], g[('wss95', 'asr_prior')], PT['P1_wss95'][1], 'post_test.json P1_wss95[1]')
out['post_hoc']['P1_wss100_hybrid_vs_pseudo'] = rep('P1 WSS@100 hybrid vs weak supervision, 23', g[('wss100', 'asr_jevblend_3')], g[('wss100', 'asr_pseudo_10_50')], PT['P1_wss100'][0], 'post_test.json P1_wss100[0]')
out['post_hoc']['P1_wss100_pseudo_vs_asr'] = rep('P1 WSS@100 weak supervision vs ASReview, 23', g[('wss100', 'asr_pseudo_10_50')], g[('wss100', 'asr_prior')], PT['P1_wss100'][1], 'post_test.json P1_wss100[1]')
out['post_hoc']['P4_wss95_h3jev_vs_h3'] = rep('P4 WSS@95 strongest preset plus Jev vs strongest preset, 23', g[('wss95', 'h3_jevblend_3')], g[('wss95', 'h3_prior')], PT['P4_wss95'][0], 'post_test.json P4_wss95[0]')
out['post_hoc']['P4_wss95_h3_vs_u4'] = rep('P4 WSS@95 strongest preset vs default, 23', g[('wss95', 'h3_prior')], g[('wss95', 'asr_prior')], PT['P4_wss95'][1], 'post_test.json P4_wss95[1]')
out['post_hoc']['P4_wss100_h3jev_vs_h3'] = rep('P4 WSS@100 strongest preset plus Jev vs strongest preset, 23', g[('wss100', 'h3_jevblend_3')], g[('wss100', 'h3_prior')], PT['P4_wss100'][0], 'post_test.json P4_wss100[0]')
At = pd.DataFrame([d for d in json.load(open(SYN + '/asr_test_ta.json')) if d.get('wss95') is not None])
gt = At.groupby(['review', 'method'])[['wss95', 'wss100']].mean().unstack('method')
out['post_hoc']['P3_ta_wss95'] = rep('P3 title-abstract labels WSS@95 hybrid vs ASReview, 12', gt[('wss95', 'asr_jevblend_3')], gt[('wss95', 'asr_prior')], PT['P3_ta_wss95'], 'post_test.json P3_ta_wss95')
out['post_hoc']['P3_ta_wss100'] = rep('P3 title-abstract labels WSS@100 hybrid vs ASReview, 12', gt[('wss100', 'asr_jevblend_3')], gt[('wss100', 'asr_prior')], PT['P3_ta_wss100'], 'post_test.json P3_ta_wss100')
Rmap = {r['id']: r for r in R}; L = lambda f: {d['id']: d['p'] for d in json.load(open(f'{SYN}/{f}')) if d.get('ok')}
S1, B10sub, B10full = L('jev_testcmp_single.json'), L('jev_testcmp_b10.json'), L('jev_test.json')
common2 = [i for i in ids if i in S1 and i in B10sub and i in B10full]; rows2 = []
for k in sorted({Rmap[i]['review'] for i in common2}):
    ii = [i for i in common2 if Rmap[i]['review'] == k]; y = [int(Rmap[i]['label']) for i in ii]
    if 0 < sum(y) < len(y): rows2.append({'review': k, 'single': roc_auc_score(y, [S1[i] for i in ii]), 'b10_subset': roc_auc_score(y, [B10sub[i] for i in ii]), 'b10_full': roc_auc_score(y, [B10full[i] for i in ii])})
P2 = pd.DataFrame(rows2)
out['post_hoc']['P2_single_minus_full'] = rep('P2 AUC single-record vs full-run batches, 23', P2['single'], P2['b10_full'], PT['P2']['single_minus_full'], 'post_test.json P2.single_minus_full')
out['post_hoc']['P2_subset_minus_full'] = rep('P2 AUC subset batches vs full-run batches, 23', P2['b10_subset'], P2['b10_full'], PT['P2']['subset_minus_full'], 'post_test.json P2.subset_minus_full')
# ---------- 7. AUC tie handling: scikit-learn roc_auc_score equals the Mann-Whitney statistic with mid-ranks (ties count one half)
def auc_midrank(y, s):
    y = np.asarray(y); s = np.asarray(s, float); r = rankdata(s, method='average'); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
chk = []
for k in revs:
    v = by[k]; y = np.array([int(r['label']) for r in v]); p = np.array([J[r['id']] for r in v])
    chk.append({'review': k, 'n_distinct_scores': int(len(np.unique(p))), 'sklearn_auc': float(roc_auc_score(y, p)), 'midrank_auc': auc_midrank(y, p)})
chk = pd.DataFrame(chk); chk['abs_diff'] = (chk.sklearn_auc - chk.midrank_auc).abs()
toy_y = [1, 1, 0, 0, 1, 0]; toy_s = [0.5, 0.5, 0.5, 0.2, 0.9, 0.9]
out['auc_tie_handling'] = {'max_abs_diff_sklearn_vs_midrank_23_heldout_reviews': float(chk.abs_diff.max()), 'toy_example': {'y': toy_y, 's': toy_s, 'sklearn': float(roc_auc_score(toy_y, toy_s)), 'midrank': auc_midrank(toy_y, toy_s)},
                           'statement': 'roc_auc_score (trapezoidal ROC) equals the Mann-Whitney U with mid-ranks; tied scores between an included and an excluded record contribute one half'}
chk.to_csv(HERE + '/auc_tie_check.csv', index=False)
# ---------- 8. code facts with line references (evidence for the Methods text)
def grep(path, pats):
    lines = open(path).read().split('\n'); hits = []
    for i, l in enumerate(lines, 1):
        if any(p in l for p in pats): hits.append({'line': i, 'text': l.strip()[:220]})
    return hits
out['code_facts'] = {
 'seed_averaging_before_review_averaging': grep(SYN + '/final_eval.py', ["groupby(['review', 'method'])", "kn = D[D.method == 'asr_prior']"]) + grep(SYN + '/clef_eval.py', ["groupby(['review', 'method'])"]) + grep(SYN + '/post_eval.py', ["groupby(['review', 'method'])"]),
 'priors_counted_in_reading_order_k95_knee': grep(SYN + '/asr_sim.py', ['def order_from_sim', "rid = [int(perm[i]) for i in sim._results", 'rid + rest', 'seq = order_from_sim', 'm = metrics(seq, y)', 'ks = knee_stop(seq, y)', 'sim.label([int(np.random.RandomState(seed).choice']),
 'knee_per_seed_then_mean_recall_then_reliability': grep(SYN + '/asr_sim.py', ['ks = knee_stop(seq, y)', "for s in range(a.seeds if m in ('asr_prior'"]) + grep(SYN + '/final_eval.py', ['knee_reliability', 'kn = D[D.method']),
 'knee_scan_grid': grep(SYN + '/al_sim.py', ['def knee_stop', 'range(150, N + 1, max(1, N // 1000))', 'rho >= 156 - min(r, 150)']),
 'metrics_definition': grep(SYN + '/al_sim.py', ['def metrics', 'k95 = pos[math.ceil(0.95 * n1) - 1]', "'wss95': 1 - k95 / N - 0.05", "rec@"]),
 'bootstrap': grep(SYN + '/final_eval.py', ['default_rng(12345)', 'def boot_diff', 'np.percentile(m, 2.5)']) + grep(SYN + '/clef_eval.py', ['default_rng(11)', 'def ci']) + grep(SYN + '/post_eval.py', ['default_rng(7)', 'def ci']),
 'wilcoxon_calls': grep(SYN + '/final_eval.py', ['wilcoxon(']) + grep(SYN + '/clef_eval.py', ['wilcoxon(']) + grep(SYN + '/post_eval.py', ['wilcoxon(']),
 'auc_sklearn': grep(SYN + '/final_eval.py', ['roc_auc_score']) [:2] + grep(SYN + '/clef_eval.py', ['roc_auc_score'])[:2],
 'clef_rank_metrics_stable_sort': grep(SYN + '/clef_eval.py', ["kind='stable'", 'def rank_metrics', 'tnr95 =', 'R@']),
 'robustness_set': {'robust_results.json reviews': RB['q1']['reviews'], 'robust_eval.py records file': grep(SYN + '/robust_eval.py', ['dev2000_records.json'])},
 'C2_rule': grep(SYN + '/final_eval.py', ["'noninferior'", "'within_0.03'", "C2 = {'pass'"]),
 'C3_C4_rules': grep(SYN + '/final_eval.py', ["'pass': bool(d.mean() >= 0.05", "c4['pass']", "'pass': bool(d.mean() >= 0.03"]),
}
# ---------- 9. development split checks used elsewhere in this bundle
out['development'] = {'n_reviews': FD['n_reviews'], 'C4_reliability': FD['C4']['reliability'], 'C4_reliability_count_implied': round(FD['C4']['reliability'] * FD['n_reviews'], 3), 'C3_wins': FD['C3']['wins'], 'C1_wins': FD['C1']['jev_wins']}
out['multiplicity'] = 'No multiplicity adjustment was applied in any original script (no such code in final_eval.py, clef_eval.py, post_eval.py); the decision rule combined four prespecified criteria with fixed thresholds and all other comparisons are descriptive.'
out['elapsed_seconds'] = round(time.time() - t0, 1); out['finished'] = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
pd.DataFrame(tests).to_csv(HERE + '/wilcoxon_table.csv', index=False); pd.DataFrame(cis).to_csv(HERE + '/ci_reproduction.csv', index=False)
out['wilcoxon_tests'] = tests; out['ci_reproduction'] = cis
json.dump(out, open(HERE + '/results.json', 'w'), indent=1, default=float)
log('CI reproduction:'); log(pd.DataFrame(cis).to_string())
log('Wilcoxon tests:'); log(pd.DataFrame(tests)[['test', 'n', 'zeros', 'ties_among_nonzero_abs_diffs', 'method_auto_inferred', 'p_auto', 'p_exact', 'p_asymptotic', 'stored_p', 'stored_matches_auto', 'p_formatted']].to_string())
log('C3 wss100:', json.dumps({k: v for k, v in out['C3'].items() if 'wss100' in k}, indent=1))
log('prereg:', json.dumps(out['prereg_comparators'], indent=1, default=float))
log('C2:', json.dumps(out['C2'], indent=1, default=float))
log('AUC ties:', json.dumps(out['auc_tie_handling'], indent=1))
log('auto rule:', out['scipy_wilcoxon_auto_rule_from_docstring'])
log('elapsed', out['elapsed_seconds'], 's')
