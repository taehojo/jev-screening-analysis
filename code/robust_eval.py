# 강건성: 질문 문구(q1,q2)와 암기 대조(제목만, 다른 리뷰 기준)를 q0(본 실행)와 비교. 리뷰별 스피어만과 거시 AUC.
import json, sys, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
R = json.load(open('dev2000_records.json')); ids = set(json.load(open('robust_ids.json')))
lab = {r['id']: r['label'] for r in R if r['id'] in ids}; rev = {r['id']: r['review'] for r in R if r['id'] in ids}
load = lambda f: {d['id']: d['p'] for d in json.load(open(f)) if d.get('ok') and d.get('p') is not None}
Q0 = load('jev_dev2000.json')
out = {}
for name, f in [('q1', 'jev_robust_q1.json'), ('q2', 'jev_robust_q2.json'), ('title_only', 'jev_robust_titleonly.json'), ('mismatch', 'jev_robust_mismatch.json')]:
    if not os.path.exists(f): continue
    V = load(f); rows = []
    for k in sorted(set(rev.values())):
        ii = [i for i in ids if rev[i] == k and i in V]
        if len(ii) < len([i for i in ids if rev[i] == k]): continue
        y = [lab[i] for i in ii]; a0 = [Q0[i] for i in ii]; a1 = [V[i] for i in ii]
        rows.append({'review': k, 'rho': spearmanr(a0, a1).correlation, 'auc_q0': roc_auc_score(y, a0), 'auc_alt': roc_auc_score(y, a1)})
    D = pd.DataFrame(rows)
    if len(D) == 0: continue
    out[name] = {'reviews': len(D), 'mean_rho': round(D.rho.mean(), 3), 'min_rho': round(D.rho.min(), 3), 'auc_q0': round(D.auc_q0.mean(), 4), 'auc_alt': round(D.auc_alt.mean(), 4), 'auc_diff': round(D.auc_alt.mean() - D.auc_q0.mean(), 4)}
    if name in ('q1', 'q2'): out[name]['pass'] = bool(D.rho.mean() >= 0.90 and abs(D.auc_alt.mean() - D.auc_q0.mean()) <= 0.02)
print(json.dumps(out, indent=1)); json.dump(out, open('robust_results.json', 'w'), indent=1)
