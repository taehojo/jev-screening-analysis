import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[_v] = '1'
# 능동학습 스크리닝 시뮬레이션 (ASReview 방식)과 Jev 혼합.
# 각 방법은 "읽는 순서"를 만든다. 지표는 그 순서에서 계산한다. 라벨은 시뮬레이션 안에서만 쓰며 외부로 나가지 않는다.
# 사용: .venv/bin/python al_sim.py <records.json> <jev_scores.json> <out.json> [--reviews k1,k2] [--seeds 10] [--procs 40] [--methods ...]
import json, argparse, math, os, sys, time
import numpy as np
from multiprocessing import Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import scipy.sparse as sp

def metrics(seq, y):
    y = np.asarray(y); N = len(y); n1 = int(y.sum())
    pos = np.where(y[np.asarray(seq)] == 1)[0] + 1  # 1-based 위치 (seq는 모든 포함을 찾을 때까지만 있어도 됨)
    if len(pos) < n1: return None
    k95 = pos[math.ceil(0.95 * n1) - 1]; k100 = pos[n1 - 1]
    out = {'N': N, 'n1': n1, 'k95': int(k95), 'k100': int(k100), 'wss95': 1 - k95 / N - 0.05, 'wss100': 1 - k100 / N}
    for f in (0.1, 0.2, 0.3):
        out[f'rec@{int(f*100)}'] = float((pos <= math.floor(f * N)).sum() / n1)
    return out


TAUS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
def knee_stop(seq, y):
    """Cormack & Grossman (2016) knee method. 150건 이상 읽은 뒤, 기울기 비 rho >= 156 - min(relret,150)이면 정지."""
    ys = y[np.asarray(seq)]; cum = np.cumsum(ys); N = len(seq)
    for s_ in range(150, N + 1, max(1, N // 1000)):
        r = cum[s_ - 1]
        if r == 0: continue
        # 무릎 i: 원점-(s,r) 직선에서 가장 먼 점
        xs = np.arange(1, s_ + 1); ys_ = cum[:s_]
        d = (r * xs - s_ * ys_) / math.hypot(r, s_)
        i = int(np.argmin(d)) + 1  # 곡선이 직선 위에 있을수록 음수
        ri = cum[i - 1]
        if i >= s_: continue
        rho = (ri / i) / ((r - ri + 1) / (s_ - i))
        if rho >= 156 - min(r, 150):
            return s_
    return N
def target_stop(seq, y, rng, T=10):
    """Cormack & Grossman (2016) target method: 무작위로 관련 T건을 찾을 때까지 표집, 그 뒤 순위대로 읽어 표적을 모두 찾으면 정지. 작업 = 표본 ∪ 읽은 앞부분."""
    N = len(y); perm = rng.permutation(N); tgt = []; sample = set()
    for i in perm:
        sample.add(int(i))
        if y[i] == 1: tgt.append(int(i))
        if len(tgt) >= T: break
    pos = {int(j): k for k, j in enumerate(seq)}
    k = max(pos[t] for t in tgt) + 1 if tgt else N
    read = set(int(j) for j in seq[:k]) | sample
    return len(read), int(sum(y[list(read)]))
def stopping(seq, y, jl, rng):
    seq = np.asarray(seq); N = len(seq); n1 = int(y.sum()); out = {}
    p = 1 / (1 + np.exp(-jl)); pos_in_seq = np.empty(N, int); pos_in_seq[seq] = np.arange(N)
    for t in TAUS:
        idx = np.where(p >= t)[0]
        k = int(pos_in_seq[idx].max()) + 1 if len(idx) else 0
        out[f'jt{t}_work'] = k / N; out[f'jt{t}_rec'] = float(y[seq[:k]].sum() / n1) if n1 else 1.0
    k = knee_stop(seq, y); out['knee_work'] = k / N; out['knee_rec'] = float(y[seq[:k]].sum() / n1)
    w, r = target_stop(seq, y, rng); out['target_work'] = w / N; out['target_rec'] = r / n1
    return out

def zs(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)

def run(task):
    X, y, jl, method, seed = task['X'], task['y'], task['jl'], task['method'], task['seed']
    N = len(y); rng = np.random.default_rng(seed); n1 = int(y.sum())
    if method.startswith('static:'):
        sc = task['static'][method.split(':', 1)[1]]
        tieb = rng.random(N) * 1e-9
        seq = list(np.argsort(-(np.asarray(sc) + tieb)))
        return {**metrics(seq, y), **stopping(seq, y, jl, rng)}
    # 형식: al|<clf>|<start>|<blend λ>|<feat w>
    _, clf_name, start, lam, fw = method.split('|'); lam = float(lam); fw = float(fw)
    labeled = np.zeros(N, bool); seq = []
    jev_order = list(np.argsort(-(jl + rng.random(N) * 1e-9))) if jl is not None else None
    if start == 'oracle':
        seq = [int(rng.choice(np.where(y == 1)[0])), int(rng.choice(np.where(y == 0)[0]))]
    else:
        src = list(rng.permutation(N)) if start == 'random' else jev_order
        for i in src:
            seq.append(int(i))
            ys = y[seq]
            if ys.max() == 1 and ys.min() == 0: break
    labeled[seq] = True
    if fw > 0:
        X = sp.hstack([X, sp.csr_matrix(fw * zs(jl).reshape(-1, 1))]).tocsr()
    b = max(1, int(round(N / 200)))
    found = int(y[seq].sum())
    while labeled.sum() < N:
        idx_l = np.where(labeled)[0]; idx_u = np.where(~labeled)[0]
        if clf_name == 'nb':
            clf = MultinomialNB(alpha=3.822)
            # 이중 균형 대신 표본 가중치로 클래스 균형
            w = np.where(y[idx_l] == 1, (len(idx_l) / (2 * max(1, y[idx_l].sum()))), (len(idx_l) / (2 * max(1, (1 - y[idx_l]).sum()))))
            clf.fit(X[idx_l], y[idx_l], sample_weight=w); s = clf.predict_log_proba(X[idx_u])[:, 1] - clf.predict_log_proba(X[idx_u])[:, 0]
        else:
            clf = LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear', max_iter=1000)
            clf.fit(X[idx_l], y[idx_l]); s = clf.decision_function(X[idx_u])
        if lam > 0: s = zs(s) + lam * zs(jl[idx_u])
        top = idx_u[np.argsort(-s)[:b]]
        for i in top:
            seq.append(int(i)); labeled[i] = True; found += int(y[i])
    return {**metrics(seq, y), **stopping(seq, y, jl, rng)}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('records'); ap.add_argument('jev'); ap.add_argument('out')
    ap.add_argument('--reviews', default=''); ap.add_argument('--seeds', type=int, default=10); ap.add_argument('--procs', type=int, default=40)
    ap.add_argument('--methods', default='static:jev,al|nb|oracle|0|0,al|lr|oracle|0|0,al|lr|random|0|0,al|lr|jev|0|0,al|lr|jev|0.5|0,al|lr|jev|1|0,al|lr|jev|2|0,al|lr|jev|0|1,al|lr|jev|0|3')
    ap.add_argument('--static', default='', help='추가 정적 점수: 이름=scores_zs 파일 패턴의 메서드명, 예: minilm')
    ap.add_argument('--zsset', default='dev')
    a = ap.parse_args()
    R = json.load(open(a.records)); J = {}
    for d in json.load(open(a.jev)):
        if d.get('ok') and d.get('p') is not None: J[d['id']] = d['p']
    by = {}
    for r in R: by.setdefault(r['review'], []).append(r)
    revs = [k for k in by if (not a.reviews or k in a.reviews.split(',')) and all(r['id'] in J for r in by[k]) and 0 < sum(r['label'] for r in by[k]) < len(by[k])]
    print(f'리뷰 {len(revs)}개 (Jev 점수가 완결된 것만)', flush=True)
    methods = a.methods.split(',')
    tasks = []
    for k in revs:
        rs = by[k]; y = np.array([r['label'] for r in rs])
        docs = [f"{r['title']}. {r['abstract']}" for r in rs]
        X = TfidfVectorizer(sublinear_tf=True, stop_words='english', ngram_range=(1, 2), min_df=1).fit_transform(docs).tocsr()
        p = np.clip(np.array([J[r['id']] for r in rs], float), 1e-4, 1 - 1e-4); jl = np.log(p / (1 - p))
        static = {'jev': jl}
        for m in [s for s in a.static.split(',') if s]:
            f = f'scores_zs/{a.zsset}_{k}.json'
            if os.path.exists(f):
                S = json.load(open(f))
                if m in S: static[m] = np.array(S[m])
        for m in methods:
            if m.startswith('static:') and m.split(':', 1)[1] not in static: continue
            nseed = 1 if m.startswith('static:') else a.seeds
            for s in range(nseed):
                tasks.append({'review': k, 'method': m, 'seed': s, 'X': X, 'y': y, 'jl': jl, 'static': static})
        for m in [s for s in a.static.split(',') if s]:
            if m in static: tasks.append({'review': k, 'method': 'static:' + m, 'seed': 0, 'X': X, 'y': y, 'jl': jl, 'static': static})
    print(f'작업 {len(tasks)}개', flush=True)
    t0 = time.time()
    with Pool(a.procs) as pool:
        res = pool.map(run, tasks, chunksize=1)
    out = [{'review': t['review'], 'method': t['method'], 'seed': t['seed'], **(r or {})} for t, r in zip(tasks, res)]
    json.dump(out, open(a.out, 'w'))
    print(f'완료 {time.time()-t0:.0f}s', flush=True)
    import pandas as pd
    df = pd.DataFrame(out)
    g = df.groupby(['review', 'method'])[['wss95', 'wss100', 'rec@10', 'rec@20']].mean().reset_index()
    print(g.groupby('method')[['wss95', 'wss100', 'rec@10', 'rec@20']].mean().sort_values('wss95', ascending=False).to_string())
