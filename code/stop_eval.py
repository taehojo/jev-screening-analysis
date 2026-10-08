# 라벨 없는 Jev 문턱 정지 규칙: Jev 확률 >= tau 인 레코드만 읽는다. 리뷰별 작업량과 재현율.
import json, sys, numpy as np, pandas as pd
recs, jevf = sys.argv[1], sys.argv[2]
R = json.load(open(recs)); J = {d['id']: d['p'] for d in json.load(open(jevf)) if d.get('ok') and d.get('p') is not None}
by = {}
for r in R: by.setdefault(r['review'], []).append(r)
TAUS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]
rows = []
for k, rs in by.items():
    if not all(r['id'] in J for r in rs): continue
    y = np.array([r['label'] for r in rs]); p = np.array([J[r['id']] for r in rs])
    if y.sum() == 0: continue
    for t in TAUS:
        sel = p >= t
        rows.append({'review': k, 'N': len(y), 'tau': t, 'work': sel.mean(), 'recall': y[sel].sum() / y.sum()})
df = pd.DataFrame(rows)
s = df.groupby('tau').agg(reviews=('review', 'nunique'), mean_work=('work', 'mean'), mean_recall=('recall', 'mean'), min_recall=('recall', 'min'),
                          reliab95=('recall', lambda r: (r >= 0.95).mean()), reliab90=('recall', lambda r: (r >= 0.90).mean()))
print(s.round(3).to_string())
