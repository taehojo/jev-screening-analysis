# ASReview 3 (기본 모델 elas_u4: TF-IDF 1-2gram + LinearSVC C=0.11 + Balanced 9.8 + Max) 기반 시뮬레이션과 Jev 혼합.
# 기준선은 ASReview 코드 그대로. 혼합은 (1) 레코드를 Jev 순으로 정렬해 시작 단계를 Jev가 고르게 하고,
# (2) SVM 점수와 Jev 로짓을 z점수로 섞는 분류기를 끼운다. 라벨은 시뮬레이션 안에서만 쓴다.
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[_v] = '1'
import json, argparse, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
from multiprocessing import Pool
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
warnings.filterwarnings('ignore')
from asreview import Simulate, ActiveLearningCycle
from asreview.models.queriers import TopDown, Max
from asreview.models.stoppers import IsFittable, LastRelevant
from asreview.models.balancers import Balanced
from asreview.models.classifiers import SVM
from asreview.models.feature_extractors import Tfidf
from al_sim import metrics, knee_stop, target_stop

def zs(v):
    v = np.asarray(v, float); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)

def _lastcol(X):
    c = X[:, -1]
    return np.asarray(c.todense()).ravel() if sp.issparse(c) else np.asarray(c).ravel()

class JevBlendSVM(BaseEstimator, ClassifierMixin):
    """ASReview의 SVM을 특징 열로만 학습하고, 순위 점수는 z(SVM) + lam * z(Jev 로짓). 마지막 열이 Jev 로짓."""
    name = 'jevblend'
    def __init__(self, lam=2.0, C=0.11, max_iter=1000):
        self.lam = lam; self.C = C; self.max_iter = max_iter
    def fit(self, X, y, sample_weight=None):
        self.svm_ = LinearSVC(C=self.C, loss='squared_hinge', max_iter=self.max_iter).fit(X[:, :-1], y, sample_weight=sample_weight); return self
    def decision_function(self, X):
        s = self.svm_.decision_function(X[:, :-1]); j = _lastcol(X)
        return zs(s) + self.lam * zs(j)


class PseudoSVM(BaseEstimator, ClassifierMixin):
    """의사 라벨 + 사람 라벨로 SVM(C=0.11)을 학습. X의 마지막 열은 레코드 번호. 균형 가중(ratio 9.8)은 합친 라벨에서 계산."""
    name = 'pseudosvm'
    def __init__(self, Xfull=None, pos=None, neg=None, C=0.11, ratio=9.8):
        self.Xfull = Xfull; self.pos = pos; self.neg = neg; self.C = C; self.ratio = ratio
    def fit(self, X, y, sample_weight=None):
        ids = set(np.asarray(X[:, -1].todense()).ravel().astype(int).tolist()) if X.shape[0] else set()
        pp = [i for i in self.pos if i not in ids]; nn = [i for i in self.neg if i not in ids]
        Xs = [X[:, :-1]] if X.shape[0] else []; ys = [np.asarray(y)] if X.shape[0] else []
        if pp: Xs.append(self.Xfull[pp][:, :-1]); ys.append(np.ones(len(pp)))
        if nn: Xs.append(self.Xfull[nn][:, :-1]); ys.append(np.zeros(len(nn)))
        Xt = sp.vstack(Xs).tocsr(); yt = np.concatenate(ys)
        if len(set(yt.tolist())) < 2:   # 한 클래스뿐이면 LLM(Jev) 순위를 그대로 따른다 (레코드 번호가 곧 Jev 순위)
            self.svm_ = None; return self
        w = np.where(yt == 1, 1.0, (yt == 1).sum() / (self.ratio * max(1, (yt == 0).sum()))); w = w * len(yt) / w.sum()
        self.svm_ = LinearSVC(C=self.C, loss='squared_hinge').fit(Xt, yt, sample_weight=w); return self
    def decision_function(self, X):
        if self.svm_ is None: return -np.asarray(X[:, -1].todense()).ravel()
        return self.svm_.decision_function(X[:, :-1])

def order_from_sim(sim, perm, N):
    rid = [int(perm[i]) for i in sim._results['record_id'].values]
    seen = set(rid); rest = [i for i in range(N) if i not in seen]   # 마지막 포함 뒤는 전부 제외 레코드
    return rid + rest

def run(task):
    k, method, seed, df, y, jl = task['review'], task['method'], task['seed'], task['df'], task['y'], task['jl']
    N = len(y); rng = np.random.RandomState(seed)
    if method in ('asr_prior', 'h3_prior'):
        perm = np.arange(N)
    elif method == 'asr_random':
        perm = rng.permutation(N)
    else:  # Jev 순서로 정렬 (동률은 무작위)
        perm = np.lexsort((rng.rand(N), -jl))
    d2 = df.iloc[perm].reset_index(drop=True); y2 = y[perm]; j2 = jl[perm]
    ratio = 9.8
    if method.startswith('h3_'):
        E2 = task['emb'][perm]; ratio = 9.724
        if method == 'h3_prior':
            X = E2; clf = SVM(C=0.067, loss='squared_hinge', max_iter=5000)
        else:
            lam = float(method.split('_')[-1]); X = np.hstack([E2, j2.reshape(-1, 1)]); clf = JevBlendSVM(lam=lam, C=0.067, max_iter=5000)
    else:
        X = sp.csr_matrix(Tfidf(ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_df=0.95).fit_transform(d2))
    if method.startswith('h3_'):
        pass
    elif method.startswith('asr_jevblend'):
        lam = float(method.split('_')[-1])
        X = sp.hstack([X, sp.csr_matrix(j2.reshape(-1, 1))]).tocsr()
        clf = JevBlendSVM(lam=lam)
    elif method.startswith('asr_pseudo'):
        _, _, T, B = method.split('_'); T = float(T) / 100; B = float(B) / 100
        X = sp.hstack([X, sp.csr_matrix(np.arange(N, dtype=float).reshape(-1, 1))]).tocsr()
        order = np.argsort(-j2, kind='stable')          # d2는 이미 Jev 순으로 정렬돼 있음
        pos = [int(i) for i in order[:max(1, int(round(T * N)))]]; neg = [int(i) for i in order[N - int(round(B * N)):]]
        clf = PseudoSVM(Xfull=X, pos=pos, neg=neg)
    elif method.startswith('asr_jevfeat'):
        w = float(method.split('_')[-1])
        X = sp.hstack([X, sp.csr_matrix((w * zs(j2)).reshape(-1, 1))]).tocsr()
        clf = SVM(C=0.11, loss='squared_hinge')
    else:
        clf = SVM(C=0.11, loss='squared_hinge')
    cycles = [ActiveLearningCycle(querier=Max(), classifier=clf, balancer=None if method.startswith('asr_pseudo') else Balanced(ratio=ratio))]
    sim = Simulate(X, y2, cycles, stopper=LastRelevant(), skip_transform=True, print_progress=False)
    if method in ('asr_prior', 'h3_prior'):
        # ASReview CLI와 같게: 포함·제외 각각 새 RandomState(seed)로 1건씩
        sim.label([int(np.random.RandomState(seed).choice(np.where(y2 == 1)[0], 1, replace=False)[0])])
        sim.label([int(np.random.RandomState(seed).choice(np.where(y2 == 0)[0], 1, replace=False)[0])])
    elif method.startswith('asr_pseudo'):
        # 의사 라벨만으로 학습한 분류기의 1순위부터 사람이 읽기 시작 (사람 라벨 0건에서 출발)
        c0 = PseudoSVM(Xfull=X, pos=pos, neg=neg).fit(X[:0], np.array([]))
        sim.label([int(np.argmax(c0.decision_function(X)))])
    else:
        # ASReview의 시작 단계(TopDown + IsFittable)와 같은 규칙: 정렬된 순서대로 읽다가 두 클래스가 모두 나오면 학습 시작
        for i in range(N):
            sim.label([i])
            if y2[:i + 1].max() == 1 and y2[:i + 1].min() == 0: break
    t0 = time.time(); sim.review()
    seq = order_from_sim(sim, perm, N)
    m = metrics(seq, y)
    if m is None: return None
    ks = knee_stop(seq, y); w, rr = target_stop(seq, y, np.random.default_rng(seed))
    m.update({'knee_work': ks / N, 'knee_rec': float(y[np.asarray(seq[:ks])].sum() / y.sum()), 'target_work': w / N, 'target_rec': rr / y.sum(), 'secs': time.time() - t0})
    m['order'] = [int(i) for i in seq]   # AN-0001-08: labelling order (record indices within the review, data-file order); written to a separate file by the main loop
    return m

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('records'); ap.add_argument('jev'); ap.add_argument('out')
    ap.add_argument('--reviews', default=''); ap.add_argument('--seeds', type=int, default=10); ap.add_argument('--procs', type=int, default=24)
    ap.add_argument('--methods', default='asr_prior,asr_random,asr_jev,asr_jevblend_1,asr_jevblend_2,asr_jevblend_3,asr_jevfeat_1')
    ap.add_argument('--max-n', type=int, default=10**9); ap.add_argument('--show', action='store_true'); ap.add_argument('--label', default='final')
    ap.add_argument('--multi-seed-methods', default='')   # npj AN-0001-03: e.g. asr_jevblend_3 (tie-breaking seeds of the Jev starting order)
    a = ap.parse_args()
    R = json.load(open(a.records)); J = {d['id']: d['p'] for d in json.load(open(a.jev)) if d.get('ok') and d.get('p') is not None} if a.jev != 'none' else None
    by = {}
    for r in R: by.setdefault(r['review'], []).append(r)
    if a.label == 'ta':
        for r in R: r['label'] = r['label_ta']
        by = {k: v for k, v in by.items() if all(r['label'] is not None for r in v)}
    revs = [k for k in by if (not a.reviews or k in a.reviews.split(',')) and len(by[k]) <= a.max_n and (J is None or all(r['id'] in J for r in by[k])) and 0 < sum(r['label'] for r in by[k]) < len(by[k])]
    done = json.load(open(a.out)) if os.path.exists(a.out) else []
    have = {(d['review'], d['method'], d['seed']) for d in done if d.get('wss95') is not None}
    tasks = []
    for k in revs:
        rs = by[k]; y = np.array([r['label'] for r in rs]); df = pd.DataFrame({'title': [r['title'] for r in rs], 'abstract': [r['abstract'] for r in rs]})
        p = np.clip(np.array([J[r['id']] for r in rs], float), 1e-4, 1 - 1e-4) if J is not None else np.full(len(rs), 0.5); jl = np.log(p / (1 - p))
        for m in a.methods.split(','):
            if m.startswith('h3_') and not os.path.exists(f'emb_mxbai/{k}.npy'): continue
            emb = np.load(f'emb_mxbai/{k}.npy') if m.startswith('h3_') else None
            for s in range(a.seeds if m in ('asr_prior', 'asr_random', 'h3_prior') or m in a.multi_seed_methods.split(',') else 1):   # npj AN-0001-03: methods listed in --multi-seed-methods also get seeds 0..seeds-1
                if (k, m, s) not in have: tasks.append({'review': k, 'method': m, 'seed': s, 'df': df, 'y': y, 'jl': jl, 'emb': emb})
    tasks.sort(key=lambda t: -len(t['y']))  # 큰 것부터 (부하 균형)
    print(f'리뷰 {len(revs)}개, 작업 {len(tasks)}개 (기존 {len(have)})', flush=True)
    out = list(done); t0 = time.time()
    orders_path = a.out[:-5] + '_orders.json' if a.out.endswith('.json') else a.out + '_orders.json'   # AN-0001-08
    orders = json.load(open(orders_path)) if os.path.exists(orders_path) else {}                       # AN-0001-08
    with Pool(a.procs) as pool:
        for i, (t, r) in enumerate(zip(tasks, pool.imap(run, tasks, chunksize=1))):
            if r is not None and 'order' in r: orders[f"{t['review']}|{t['method']}|{t['seed']}"] = r.pop('order')   # AN-0001-08
            out.append({'review': t['review'], 'method': t['method'], 'seed': t['seed'], **(r or {})})
            if (i + 1) % 50 == 0 or i + 1 == len(tasks):
                json.dump(out, open(a.out, 'w')); json.dump(orders, open(orders_path, 'w')); print(f'  {i+1}/{len(tasks)} {time.time()-t0:.0f}s', flush=True)   # AN-0001-08: orders file added
    if 'test' in a.out and not a.show:
        print('평가 분할: 요약 출력 생략 (판정 전)'); raise SystemExit
    df = pd.DataFrame([o for o in out if o.get('wss95') is not None])
    g = df.groupby(['review', 'method'])[['wss95', 'wss100', 'rec@10', 'rec@20', 'knee_work', 'knee_rec']].mean().reset_index()
    print(g.groupby('method')[['wss95', 'wss100', 'rec@10', 'rec@20', 'knee_work', 'knee_rec']].mean().sort_values('wss95', ascending=False).to_string())
