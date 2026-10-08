# CLEF TAR 2019 평가(코크란 31편): 제로샷 순위 지표(LGAR와 같은 정의), 능동학습 시뮬레이션, 라벨 없는 정지 규칙
import json, os, math, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
rng = np.random.default_rng(11)
def ci(d, B=5000):
    d = np.asarray(d); m = d[rng.integers(0, len(d), (B, len(d)))].mean(1); return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]
def rank_metrics(y, s):
    y = np.asarray(y); o = np.argsort(-np.asarray(s), kind='stable'); ys = y[o]; n1 = ys.sum(); N = len(y)
    prec = np.cumsum(ys) / np.arange(1, N + 1); ap = float((prec * ys).sum() / n1)
    k95 = int(np.searchsorted(np.cumsum(ys), math.ceil(0.95 * n1)) + 1)
    tnr95 = ((N - k95) - (n1 - ys[:k95].sum())) / (N - n1)
    rk = {f'R@{p}%': float(ys[:max(1, int(round(p / 100 * N)))].sum() / n1) for p in (1, 5, 10, 20, 50)}
    return {'AUC': roc_auc_score(y, s), 'MAP': ap, 'TNR@95': float(tnr95), **rk}
R = json.load(open('../clef/clef_records.json')); J = {d['id']: d['p'] for d in json.load(open('jev_clef.json')) if d.get('ok')}
by = {}
for r in R: by.setdefault(r['review'], []).append(r)
out = {}
for lab in ['label', 'label_ta']:
    rows = []
    for k, v in by.items():
        if not all(r['id'] in J for r in v): continue
        y = np.array([r[lab] or 0 for r in v])
        if not (0 < y.sum() < len(y)): continue
        row = {'topic': k, 'type': v[0]['split'], 'N': len(v), 'n1': int(y.sum()), **{f'jev_{m}': x for m, x in rank_metrics(y, [J[r['id']] for r in v]).items()}}
        f = f'scores_zs/clef_{k}.json'
        if os.path.exists(f):
            S = json.load(open(f))
            for meth in ['bge', 'minilm', 'bm25']:
                if meth in S: row.update({f'{meth}_{m}': x for m, x in rank_metrics(y, S[meth]).items()})
        rows.append(row)
    T = pd.DataFrame(rows); T.to_csv(f'clef_zs_{lab}.csv', index=False)
    res = {'topics': len(T), 'jev': {m: round(float(T[f'jev_{m}'].mean()), 4) for m in ['AUC', 'MAP', 'TNR@95', 'R@10%', 'R@20%']}}
    if 'bge_AUC' in T:
        res['bge'] = {m: round(float(T[f'bge_{m}'].mean()), 4) for m in ['AUC', 'MAP', 'TNR@95', 'R@10%', 'R@20%']}
        d = T.jev_AUC - T.bge_AUC; res['jev_vs_bge_auc'] = {'diff': round(float(d.mean()), 4), 'ci': ci(d), 'wins': int((d > 0).sum()), 'p': float(wilcoxon(T.jev_AUC, T.bge_AUC).pvalue)}
    res['by_type_jev_auc'] = T.groupby('type').jev_AUC.mean().round(4).to_dict()
    out[f'zeroshot_{lab}'] = res
# 능동학습 시뮬레이션
if os.path.exists('asr_clef.json'):
    A = pd.DataFrame([d for d in json.load(open('asr_clef.json')) if d.get('wss95') is not None])
    g = A.groupby(['review', 'method'])[['wss95', 'wss100', 'knee_work', 'knee_rec']].mean().unstack('method')
    out['al_wss95'] = {m: round(float(g[('wss95', m)].mean()), 4) for m in g['wss95'].columns}
    out['al_wss100'] = {m: round(float(g[('wss100', m)].mean()), 4) for m in g['wss100'].columns}
    for a_, b_ in [('asr_jevblend_3', 'asr_prior'), ('asr_jevblend_3', 'asr_pseudo_10_50')]:
        if ('wss95', a_) in g and ('wss95', b_) in g:
            for met in ['wss95', 'wss100']:
                x, z = g[(met, a_)], g[(met, b_)]; ok = x.notna() & z.notna(); d = (x - z)[ok]
                out[f'{met}_{a_}_vs_{b_}'] = {'n': int(ok.sum()), 'diff': round(float(d.mean()), 4), 'ci': ci(d), 'wins': int((d > 0).sum()), 'p': float(wilcoxon(x[ok], z[ok]).pvalue)}
    kn = g[('knee_rec', 'asr_prior')]; out['knee_asreview'] = {'reliability': round(float((kn >= 0.95).mean()), 4), 'work': round(float(g[('knee_work', 'asr_prior')].mean()), 4)}
# 라벨 없는 정지 (τ=0.07, SYNERGY 개발 분할에서 고정)
rec, work = [], []
for k, v in by.items():
    if not all(r['id'] in J for r in v): continue
    y = np.array([r['label'] for r in v]); p = np.array([J[r['id']] for r in v]); s = p >= 0.07
    if y.sum(): rec.append(y[s].sum() / y.sum()); work.append(s.mean())
rec, work = np.array(rec), np.array(work)
out['stop_tau0.07'] = {'topics': len(rec), 'reliability95': round(float((rec >= 0.95).mean()), 4), 'mean_recall': round(float(rec.mean()), 4), 'min_recall': round(float(rec.min()), 4), 'mean_work': round(float(work.mean()), 4)}
out['LGAR_reference_TAR2019'] = {'MAP': 0.506, 'TNR@95': 0.765, 'R@10%': 0.767, 'R@20%': 0.883, 'source': 'arXiv 2505.24757 Table (TAR2019 test, 31 SLRs)'}
print(json.dumps(out, indent=1, default=float)); json.dump(out, open('clef_results.json', 'w'), indent=1, default=float)
