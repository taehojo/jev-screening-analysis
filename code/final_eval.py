# 사전 등록 판정 C1~C4를 계산한다. 사용: .venv/bin/python final_eval.py <dev|test> [--best-free minilm] [--lam 2] [--tau 0.05]
import json, argparse, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
ap = argparse.ArgumentParser(); ap.add_argument('split'); ap.add_argument('--best-free', default=None); ap.add_argument('--lam', default='2'); ap.add_argument('--tau', type=float, default=None)
a = ap.parse_args()
REC = {'dev': 'dev2000_records.json', 'test': 'test_records.json'}[a.split]; JEV = {'dev': 'jev_dev2000.json', 'test': 'jev_test.json'}[a.split]
R = json.load(open(REC)); J = {d['id']: d['p'] for d in json.load(open(JEV)) if d.get('ok') and d.get('p') is not None}
by = {}
for r in R: by.setdefault(r['review'], []).append(r)
revs = [k for k, v in by.items() if all(r['id'] in J for r in v) and 0 < sum(r['label'] for r in v) < len(v)]
rng = np.random.default_rng(12345)
def boot_diff(d, B=5000):
    d = np.asarray(d); idx = rng.integers(0, len(d), (B, len(d))); m = d[idx].mean(1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
out = {'split': a.split, 'n_reviews': len(revs)}
# ---------- C1: 제로샷 AUC (전체 레코드)
rows = []
for k in revs:
    v = by[k]; y = np.array([r['label'] for r in v]); row = {'review': k, 'n': len(v), 'n1': int(y.sum()), 'jev': roc_auc_score(y, [J[r['id']] for r in v])}
    f = f'scores_zs/{a.split}_{k}.json'
    if os.path.exists(f):
        S = json.load(open(f))
        for m, s in S.items(): row[m] = roc_auc_score(y, s)
    rows.append(row)
A = pd.DataFrame(rows); free = [c for c in A.columns if c not in ('review', 'n', 'n1', 'jev')]
out['C1_macro_auc'] = {m: round(float(A[m].mean()), 4) for m in ['jev'] + free if m in A and A[m].notna().all()}
best = a.best_free or (max(free, key=lambda m: A[m].mean()) if free else None)
if best and A[best].notna().all():
    d = A['jev'] - A[best]; lo, hi = boot_diff(d)
    p = wilcoxon(A['jev'], A[best]).pvalue if len(d) > 5 else None
    out['C1'] = {'best_free': best, 'diff': round(float(d.mean()), 4), 'ci': [round(lo, 4), round(hi, 4)], 'wilcoxon_p': p, 'jev_wins': int((d > 0).sum()), 'n': len(d),
                 'pass': bool(d.mean() >= 0.03 and p is not None and p < 0.01)}
# ---------- C2: LLM 비교 (평가 부분집합)
if a.split == 'test' and os.path.exists('test_cmp_ids.json'):
    ids = set(json.load(open('test_cmp_ids.json'))); lab = {r['id']: r['label'] for r in R}; rev = {r['id']: r['review'] for r in R}
    models = {'jev': J}
    for name, f in [('gpt4omini_lp', 'test_cmp_gpt4omini_lp.json'), ('deepseek', 'test_cmp_deepseek.json'), ('claude_opus', 'test_cmp_claude-opus.json')]:
        if os.path.exists(f): models[name] = {d['id']: d['p'] for d in json.load(open(f)) if d.get('ok') and d.get('p') is not None}
    common = [i for i in ids if all(i in M for M in models.values())]
    out['C2_n_common'] = len(common)
    rr = []
    for k in sorted({rev[i] for i in common}):
        ii = [i for i in common if rev[i] == k]; y = np.array([lab[i] for i in ii])
        if 0 < y.sum() < len(y): rr.append({'review': k, **{m: roc_auc_score(y, [M[i] for i in ii]) for m, M in models.items()}})
    B2 = pd.DataFrame(rr); out['C2_macro_auc'] = {m: round(float(B2[m].mean()), 4) for m in models}
    if 'gpt4omini_lp' in B2:
        d = B2['jev'] - B2['gpt4omini_lp']; lo, hi = boot_diff(d); out['C2_vs_gpt4omini'] = {'diff': round(float(d.mean()), 4), 'ci': [round(lo, 4), round(hi, 4)], 'noninferior': bool(lo > -0.02)}
    if 'claude_opus' in B2:
        d = B2['claude_opus'] - B2['jev']; lo, hi = boot_diff(d); out['C2_vs_claude'] = {'claude_minus_jev': round(float(d.mean()), 4), 'ci': [round(lo, 4), round(hi, 4)], 'within_0.03': bool(d.mean() <= 0.03)}
    if 'deepseek' in B2:
        d = B2['jev'] - B2['deepseek']; lo, hi = boot_diff(d); out['C2_vs_deepseek'] = {'diff': round(float(d.mean()), 4), 'ci': [round(lo, 4), round(hi, 4)]}
    out['C2'] = {'pass': bool(out.get('C2_vs_gpt4omini', {}).get('noninferior') and out.get('C2_vs_claude', {}).get('within_0.03'))}
# ---------- C3: 혼합 대 ASReview 기본
f = f'asr_{a.split}.json'
if os.path.exists(f):
    D = pd.DataFrame([d for d in json.load(open(f)) if d.get('wss95') is not None and d['review'] in revs])
    g = D.groupby(['review', 'method'])[['wss95', 'wss100', 'knee_work', 'knee_rec']].mean().unstack('method')
    hy = f'asr_jevblend_{a.lam}'
    if ('wss95', hy) in g and ('wss95', 'asr_prior') in g:
        x = g[('wss95', hy)]; b = g[('wss95', 'asr_prior')]; ok = x.notna() & b.notna(); d = (x - b)[ok]; lo, hi = boot_diff(d)
        d100 = (g[('wss100', hy)] - g[('wss100', 'asr_prior')])[ok]
        out['C3'] = {'hybrid': hy, 'n': int(ok.sum()), 'wss95_hybrid': round(float(x[ok].mean()), 4), 'wss95_asr': round(float(b[ok].mean()), 4), 'diff': round(float(d.mean()), 4), 'ci': [round(lo, 4), round(hi, 4)],
                     'wilcoxon_p': float(wilcoxon(x[ok], b[ok]).pvalue), 'wins': int((d > 0).sum()), 'wss100_diff': round(float(d100.mean()), 4),
                     'pass': bool(d.mean() >= 0.05 and wilcoxon(x[ok], b[ok]).pvalue < 0.05 and d100.mean() > 0)}
        out['C3_all_methods_wss95'] = {m: round(float(g[('wss95', m)].mean()), 4) for m in g['wss95'].columns}
        out['C3_all_methods_wss100'] = {m: round(float(g[('wss100', m)].mean()), 4) for m in g['wss100'].columns}
# ---------- C4: 라벨 없는 Jev 문턱 규칙 대 knee(ASReview 기본 순서)
if a.tau is not None:
    rec, work = [], []
    for k in revs:
        v = by[k]; y = np.array([r['label'] for r in v]); p = np.array([J[r['id']] for r in v]); s = p >= a.tau
        rec.append(y[s].sum() / y.sum()); work.append(s.mean())
    rec = np.array(rec); work = np.array(work)
    c4 = {'tau': a.tau, 'reliability': round(float((rec >= 0.95).mean()), 4), 'mean_work': round(float(work.mean()), 4), 'mean_recall': round(float(rec.mean()), 4), 'min_recall': round(float(rec.min()), 4)}
    if os.path.exists(f):
        kn = D[D.method == 'asr_prior'].groupby('review')[['knee_work', 'knee_rec']].mean().reindex(revs)
        c4.update({'knee_reliability': round(float((kn.knee_rec >= 0.95).mean()), 4), 'knee_mean_work': round(float(kn.knee_work.mean()), 4)})
        c4['pass'] = bool(c4['reliability'] >= 0.90 and c4['reliability'] >= c4['knee_reliability'] and c4['mean_work'] < c4['knee_mean_work'])
    out['C4'] = c4
print(json.dumps(out, indent=1, default=float))
json.dump(out, open(f'final_{a.split}.json', 'w'), indent=1, default=float)
