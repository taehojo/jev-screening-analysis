# 사후 분석 요약: P1(의사 라벨), P2(묶음 효과), P3(초록 단계 라벨), P4(elas_h3) — 평가 분할 기준
import json, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon, spearmanr
rng = np.random.default_rng(7)
def ci(d, B=5000):
    d = np.asarray(d); m = d[rng.integers(0, len(d), (B, len(d)))].mean(1); return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]
def paired(a, b, name):
    ok = a.notna() & b.notna(); d = (a - b)[ok]
    return {'compare': name, 'n': int(ok.sum()), 'mean_a': round(float(a[ok].mean()), 4), 'mean_b': round(float(b[ok].mean()), 4), 'diff': round(float(d.mean()), 4), 'ci': ci(d), 'wins': int((d > 0).sum()), 'losses': int((d < 0).sum()), 'wilcoxon_p': float(wilcoxon(a[ok], b[ok]).pvalue) if ok.sum() > 5 and (d != 0).any() else None}
out = {}
def wss_table(f):
    A = pd.DataFrame([d for d in json.load(open(f)) if d.get('wss95') is not None])
    return A.groupby(['review', 'method'])[['wss95', 'wss100', 'knee_work', 'knee_rec']].mean().unstack('method')
# P1
g = wss_table('asr_test.json')
if ('wss95', 'asr_pseudo_10_50') in g:
    out['P1_wss95'] = [paired(g[('wss95', 'asr_jevblend_3')], g[('wss95', 'asr_pseudo_10_50')], 'hybrid λ3 vs pseudo-label 10/50'), paired(g[('wss95', 'asr_pseudo_10_50')], g[('wss95', 'asr_prior')], 'pseudo-label vs ASReview')]
    out['P1_wss100'] = [paired(g[('wss100', 'asr_jevblend_3')], g[('wss100', 'asr_pseudo_10_50')], 'hybrid vs pseudo'), paired(g[('wss100', 'asr_pseudo_10_50')], g[('wss100', 'asr_prior')], 'pseudo vs ASReview')]
# P4
if ('wss95', 'h3_prior') in g and g[('wss95', 'h3_prior')].notna().sum() > 5:
    out['P4_wss95'] = [paired(g[('wss95', 'h3_jevblend_3')], g[('wss95', 'h3_prior')], 'h3+Jev vs ASReview elas_h3'), paired(g[('wss95', 'h3_prior')], g[('wss95', 'asr_prior')], 'elas_h3 vs elas_u4')]
    out['P4_wss100'] = [paired(g[('wss100', 'h3_jevblend_3')], g[('wss100', 'h3_prior')], 'h3+Jev vs elas_h3')]
# P3
if os.path.exists('asr_test_ta.json'):
    t = wss_table('asr_test_ta.json')
    out['P3_ta_wss95'] = paired(t[('wss95', 'asr_jevblend_3')], t[('wss95', 'asr_prior')], 'TA labels: hybrid vs ASReview')
    out['P3_ta_wss100'] = paired(t[('wss100', 'asr_jevblend_3')], t[('wss100', 'asr_prior')], 'TA labels: hybrid vs ASReview')
# P2
if os.path.exists('jev_testcmp_single.json') and os.path.exists('jev_testcmp_b10.json'):
    R = {r['id']: r for r in json.load(open('test_records.json'))}; ids = set(json.load(open('test_cmp_ids.json')))
    L = lambda f: {d['id']: d['p'] for d in json.load(open(f)) if d.get('ok')}
    S1, B10sub, B10full = L('jev_testcmp_single.json'), L('jev_testcmp_b10.json'), L('jev_test.json')
    common = [i for i in ids if i in S1 and i in B10sub and i in B10full]; rows = []
    for k in sorted({R[i]['review'] for i in common}):
        ii = [i for i in common if R[i]['review'] == k]; y = [R[i]['label'] for i in ii]
        if 0 < sum(y) < len(y):
            rows.append({'review': k, 'single': roc_auc_score(y, [S1[i] for i in ii]), 'b10_subset': roc_auc_score(y, [B10sub[i] for i in ii]), 'b10_full': roc_auc_score(y, [B10full[i] for i in ii]),
                         'rho_single_full': spearmanr([S1[i] for i in ii], [B10full[i] for i in ii]).correlation, 'rho_sub_full': spearmanr([B10sub[i] for i in ii], [B10full[i] for i in ii]).correlation})
    P = pd.DataFrame(rows)
    y = np.array([R[i]['label'] for i in common]); prev_sub = float(y.mean())
    out['P2'] = {'n_records': len(common), 'reviews': len(P), 'macro_auc': {c: round(float(P[c].mean()), 4) for c in ['single', 'b10_subset', 'b10_full']},
                 'single_minus_full': paired(P['single'], P['b10_full'], 'single vs batch10(full, prevalence ~1.8%)'), 'subset_minus_full': paired(P['b10_subset'], P['b10_full'], 'batch10 subset (prevalence ~30%) vs batch10 full'),
                 'mean_rho_single_full': round(float(P.rho_single_full.mean()), 3), 'mean_rho_sub_full': round(float(P.rho_sub_full.mean()), 3),
                 'mean_p_pos': {n: round(float(np.mean([D[i] for i in common if R[i]['label'] == 1])), 3) for n, D in [('single', S1), ('b10_subset', B10sub), ('b10_full', B10full)]},
                 'mean_p_neg': {n: round(float(np.mean([D[i] for i in common if R[i]['label'] == 0])), 3) for n, D in [('single', S1), ('b10_subset', B10sub), ('b10_full', B10full)]}, 'subset_prevalence': round(prev_sub, 3)}
print(json.dumps(out, indent=1, default=float)); json.dump(out, open('post_test.json', 'w'), indent=1, default=float)
