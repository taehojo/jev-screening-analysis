# 무료 제로샷 기준선: 적격 기준 문단을 쿼리로, 제목+초록을 문서로 놓고 관련도 점수를 매긴다. 전부 로컬, 비용 0.
# 사용: .venv/bin/python zs_baselines.py <dhl|dev|test> [--methods tfidf,bm25,minilm,bge,e5,ce_bge,ce_m3] [--reviews k1,k2] [--max-n 2000]
import json, sys, os, re, time, argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
p = argparse.ArgumentParser()
p.add_argument('set'); p.add_argument('--methods', default='tfidf,bm25,minilm,bge,e5,ce_bge')
p.add_argument('--reviews', default=''); p.add_argument('--max-n', type=int, default=10**9); p.add_argument('--threads', type=int, default=32)
a = p.parse_args()
import torch; torch.set_num_threads(a.threads)
os.makedirs('scores_zs', exist_ok=True)
if a.set == 'dhl':
    R = json.load(open('dhl_records.json')); C = json.load(open('dhl_criteria.json'))
elif a.set == 'clef':
    R = json.load(open('../clef/clef_records.json')); C = json.load(open('../clef/clef_criteria.json'))
else:
    R = json.load(open({'dev': 'dev2000_records.json', 'test': 'test_records.json', 'all': 'records.json'}[a.set])); C = json.load(open('criteria.json'))
by = {}
for r in R: by.setdefault(r['review'], []).append(r)
revs = [k for k in by if (not a.reviews or k in a.reviews.split(',')) and len(by[k]) <= a.max_n]
methods = a.methods.split(',')
STOP = set('a an the of and or in on for to with by from at as is are was were be been this that these those it its their we our which who whom whose not no any all into than then there such can may must should studies study'.split())
tok = lambda s: [w for w in re.findall(r'[a-z0-9]+', s.lower()) if w not in STOP and len(w) > 1]
models = {}
def st(name):
    from sentence_transformers import SentenceTransformer
    if name not in models: models[name] = SentenceTransformer(name, device='cpu')
    return models[name]
def ce(name, maxlen):
    from sentence_transformers import CrossEncoder
    if name not in models: models[name] = CrossEncoder(name, device='cpu', max_length=maxlen)
    return models[name]
DENSE = {'minilm': ('sentence-transformers/all-MiniLM-L6-v2', '', ''),
         'bge': ('BAAI/bge-base-en-v1.5', 'Represent this sentence for searching relevant passages: ', ''),
         'e5': ('intfloat/e5-base-v2', 'query: ', 'passage: ')}
CROSS = {'ce_bge': ('BAAI/bge-reranker-base', 512), 'ce_m3': ('BAAI/bge-reranker-v2-m3', 1024)}
summary = []
for k in revs:
    recs = by[k]; q = C[k]; docs = [f"{r['title']}. {r['abstract']}" for r in recs]; y = np.array([r['label'] for r in recs])
    out_path = f'scores_zs/{a.set}_{k}.json'
    S = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for m in methods:
        if m in S: continue
        t0 = time.time()
        if m == 'tfidf':
            v = TfidfVectorizer(sublinear_tf=True, stop_words='english', ngram_range=(1, 2), min_df=1).fit(docs + [q])
            s = (v.transform(docs) @ v.transform([q]).T).toarray().ravel()
        elif m == 'bm25':
            from rank_bm25 import BM25Okapi
            s = BM25Okapi([tok(d) for d in docs]).get_scores(tok(q))
        elif m in DENSE:
            name, qp, dp = DENSE[m]; mod = st(name)
            qe = mod.encode([qp + q], normalize_embeddings=True); de = mod.encode([dp + d for d in docs], batch_size=64, normalize_embeddings=True)
            s = (de @ qe.T).ravel()
        elif m in CROSS:
            name, ml = CROSS[m]; mod = ce(name, ml)
            s = np.asarray(mod.predict([(q, d) for d in docs], batch_size=16, show_progress_bar=False)).ravel()
        S[m] = [float(x) for x in s]
        json.dump(S, open(out_path, 'w'))
        print(f'{k} n={len(recs)} {m}: {time.time()-t0:.0f}s', flush=True)
    row = {'review': k, 'n': len(recs), 'incl': int(y.sum())}
    if 0 < y.sum() < len(y):
        for m in methods: row[m] = roc_auc_score(y, S[m])
    summary.append(row)
import pandas as pd
df = pd.DataFrame(summary); df.to_csv(f'zs_auc_{a.set}.csv', index=False)
print(df.to_string()); print('macro AUC:', {m: round(df[m].mean(), 4) for m in methods if m in df})
