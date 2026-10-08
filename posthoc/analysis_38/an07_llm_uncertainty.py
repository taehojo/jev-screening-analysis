# AN-0001-07 (npj round 1): LLM comparison on the 2002-record held-out subset.
# (a) two-stage bootstrap (reviews, then excluded records within review) of macro AUC and differences;
# (b) sensitivity of the Claude Opus 5.5 values to the three refused records (score 0 or 1);
# (c) calibration of each model's scores on the (enriched) subset at both label levels.
# Post hoc. Reads stored outputs only; no model call. C2 as reported is not changed.
# Calibration functions logit, sigmoid, fit_logit (two-parameter Newton branch), ece_equal_width are copied from
# review_pipeline/rounds/round_0001/revision/analysis/AN-0001-01/calibration.py (Lancet AN-0001-01), unchanged.
import json, datetime, time
from fractions import Fraction as F
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
S = '/N/project/AiLab/jev/synergy/'
SEED = 20260928; B = 5000; B_CAL = 1000; EPS = 1e-4
T0 = time.time()
# ---------- copied from Lancet AN-0001-01 calibration.py ----------
def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS); return np.log(p / (1 - p))
def sigmoid(z): return 1 / (1 + np.exp(-z))
def fit_logit(x, y):
    X = np.column_stack([np.ones(len(x)), x]); beta = np.zeros(2); conv = False; it = 0
    for it in range(1, 101):
        eta = X @ beta; mu = sigmoid(eta); w = mu * (1 - mu); g = X.T @ (y - mu); H = (X * w[:, None]).T @ X
        try: step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError: break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10: conv = True; break
    if abs(beta[1]) > 50 or not np.all(np.isfinite(beta)): conv = False
    return float(beta[0]), float(beta[1]), conv, it
def ece_equal_width(p, y, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1); e = 0.0
    for i in range(nb):
        m = b == i
        if m.sum() == 0: continue
        e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)
# -------------------------------------------------------------------
R = json.load(open(S + 'test_records.json'))
ids = json.load(open(S + 'test_cmp_ids.json')); idset = set(ids)
rec = {r['id']: r for r in R if r['id'] in idset}
def load(f):
    return {d['id']: d['p'] for d in json.load(open(S + f)) if d.get('ok') and d.get('p') is not None}
M = {'jev': load('jev_test.json'), 'gpt4omini_lp': load('test_cmp_gpt4omini_lp.json'), 'deepseek': load('test_cmp_deepseek.json'), 'claude_opus': load('test_cmp_claude-opus.json')}
models = list(M)
common = [i for i in ids if all(i in M[m] for m in models)]
refused = [i for i in ids if i not in M['claude_opus']]
out = {'analysis_id': 'AN-0001-07', 'seed': SEED, 'n_subset': len(ids), 'n_common': len(common), 'refused_claude_ids_count': len(refused),
       'refused_claude': [{'review': rec[i]['review'], 'label': int(rec[i]['label'])} for i in refused]}
def auc_exact(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float); n1 = int(y.sum()); n0 = len(y) - n1
    r2 = np.rint(2 * rankdata(s, method='average')).astype(np.int64); u2 = int(r2[y == 1].sum()) - n1 * (n1 + 1)
    return F(u2, 2 * n1 * n0)
def by_review(idl):
    g = {}
    for i in idl: g.setdefault(rec[i]['review'], []).append(i)
    return g
# ---------- (a) two-stage bootstrap ----------
G = by_review(common); revs = sorted(G)
# per review and model: contribution c_j of each excluded record j = (#pos with s > s_j + 0.5 #pos with s == s_j) / n_pos
C = {}; per = []
for k in revs:
    ii = G[k]; y = np.array([int(rec[i]['label']) for i in ii]); row = {'review': k, 'n_pos': int(y.sum()), 'n_neg': int((1 - y).sum())}
    C[k] = {}
    for m in models:
        s = np.array([M[m][i] for i in ii], float); sp = s[y == 1]; sn = s[y == 0]
        c = np.array([((sp > v).sum() + 0.5 * (sp == v).sum()) / len(sp) for v in sn])
        C[k][m] = c; row[m] = float(c.mean())
        assert abs(row[m] - roc_auc_score(y, s)) < 1e-12
    per.append(row)
out['macro_auc_point'] = {m: float(np.mean([r[m] for r in per])) for m in models}
diffs = {'claude_minus_jev': ('claude_opus', 'jev'), 'jev_minus_gpt4omini': ('jev', 'gpt4omini_lp'), 'jev_minus_deepseek': ('jev', 'deepseek')}
rng = np.random.default_rng(SEED)
nrev = len(revs)
two = {m: np.empty(B) for m in models}; one = {m: np.empty(B) for m in models}
IDX = rng.integers(0, nrev, (B, nrev))
per_auc = {m: np.array([r[m] for r in per]) for m in models}
for b in range(B):
    acc = {m: 0.0 for m in models}
    for j in IDX[b]:
        k = revs[j]; nn = len(C[k]['jev'])
        w = rng.integers(0, nn, nn)          # resample excluded records of this review (same draw for all models)
        for m in models: acc[m] += C[k][m][w].mean()
    for m in models:
        two[m][b] = acc[m] / nrev; one[m][b] = per_auc[m][IDX[b]].mean()
pc = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
out['two_stage'] = {'B': B, 'seed': SEED, 'description': 'stage 1: 23 reviews with replacement; stage 2: within each drawn review, its excluded records with replacement (included records kept); the same excluded-record draw is used for all four models; macro AUC = mean over drawn reviews',
                    'macro_auc_ci': {m: pc(two[m]) for m in models},
                    'diff': {d: {'point': out['macro_auc_point'][a] - out['macro_auc_point'][b_], 'ci_two_stage': pc(two[a] - two[b_]), 'ci_one_stage_same_review_draws': pc(one[a] - one[b_]),
                                 'proportion_of_two_stage_resamples_at_or_below_zero': float(np.mean((two[a] - two[b_]) <= 0))} for d, (a, b_) in diffs.items()},
                    'one_stage_macro_auc_ci': {m: pc(one[m]) for m in models}}
for d in out['two_stage']['diff'].values():
    d['width_two_stage'] = d['ci_two_stage'][1] - d['ci_two_stage'][0]; d['width_one_stage'] = d['ci_one_stage_same_review_draws'][1] - d['ci_one_stage_same_review_draws'][0]
# ---------- (b) refused records ----------
G2 = by_review(ids); revs2 = sorted(G2); sens = {}
rng_b = np.random.default_rng(SEED); IDX2 = rng_b.integers(0, len(revs2), (B, len(revs2)))
for fill in (0.0, 1.0):
    rows = []
    for k in revs2:
        ii = G2[k]; y = [int(rec[i]['label']) for i in ii]
        cl = [M['claude_opus'].get(i, fill) for i in ii]; jv = [M['jev'][i] for i in ii]
        rows.append({'review': k, 'claude': float(auc_exact(y, cl)), 'jev': float(auc_exact(y, jv))})
    cl = np.array([r['claude'] for r in rows]); jv = np.array([r['jev'] for r in rows]); d = cl - jv
    sens[f'refused_scored_{int(fill)}'] = {'n_records': len(ids), 'macro_auc_claude': float(cl.mean()), 'macro_auc_jev': float(jv.mean()), 'claude_minus_jev': float(d.mean()),
                                           'ci_one_stage': pc(d[IDX2].mean(1)), 'claude_higher_reviews': int((d > 0).sum()), 'claude_lower_reviews': int((d < 0).sum()), 'tied_reviews': int((d == 0).sum()),
                                           'reviews_changed_vs_common_set': [r['review'] for r in rows if r['review'] in ('Attai_2022', 'Walker_2018')],
                                           'per_review_affected': [r for r in rows if r['review'] in ('Attai_2022', 'Walker_2018')]}
out['refused_sensitivity'] = sens
out['refused_sensitivity_note'] = 'All 2002 records; the three records refused by the Claude Opus 5.5 interface are given score 0 (lowest possible) or 1 (ranked first, as if routed to humans). Jev on the same 2002 records. One-stage review bootstrap, B=5000, seed 20260928.'
# ---------- (c) calibration ----------
def cal_core(p, y):
    lp = logit(p); a, b_, conv, it = fit_logit(lp, y)
    return {'n': int(len(y)), 'n_pos': int(y.sum()), 'observed_rate': float(y.mean()), 'mean_predicted': float(p.mean()), 'O_E_ratio': float(y.sum() / p.sum()),
            'recal_intercept': a, 'recal_slope': b_, 'recal_converged': conv, 'brier': float(np.mean((p - y) ** 2)), 'ece_10_equal_width': ece_equal_width(p, y)}
cal = {}
for level in ('final', 'ta'):
    key = 'label' if level == 'final' else 'label_ta'
    sel = [i for i in common if rec[i].get(key) not in (None, '')]
    Gc = by_review(sel); rv = sorted(Gc); cal[level] = {'n_reviews': len(rv), 'n_records': len(sel)}
    rng_c = np.random.default_rng(SEED)
    IDXc = rng_c.integers(0, len(rv), (B_CAL, len(rv)))
    for m in models:
        p = np.array([M[m][i] for i in sel], float); y = np.array([int(rec[i][key]) for i in sel])
        res = cal_core(p, y)
        res['n_scores_clipped_by_1e-4'] = int(((p < EPS) | (p > 1 - EPS)).sum())
        grp = {k: (np.array([M[m][i] for i in Gc[k]], float), np.array([int(rec[i][key]) for i in Gc[k]])) for k in rv}
        Bv = {'O_E_ratio': [], 'recal_intercept': [], 'recal_slope': [], 'brier': [], 'ece_10_equal_width': []}; nconv = 0
        for b in range(B_CAL):
            pp = np.concatenate([grp[rv[j]][0] for j in IDXc[b]]); yy = np.concatenate([grp[rv[j]][1] for j in IDXc[b]])
            if yy.sum() == 0 or yy.sum() == len(yy): continue
            r_ = cal_core(pp, yy)
            if not r_['recal_converged']: nconv += 1; continue
            for q in Bv: Bv[q].append(r_[q])
        res['bootstrap_ci_reviews'] = {q: pc(v) for q, v in Bv.items()}; res['bootstrap_ci_reviews']['B_used'] = len(Bv['recal_slope']); res['bootstrap_ci_reviews']['not_converged_dropped'] = nconv
        res['n_distinct_scores'] = int(len(np.unique(p)))
        cal[level][m] = res
out['calibration'] = cal
out['calibration_note'] = ('The subset holds all included records and about 4.3% of the excluded records per review (29.8% included at the final level), so O:E and '
                           'the calibration intercept describe the enriched subset, not a full search. The slope and the ranking-related quantities are less affected by the sampling of excluded records. '
                           'Scores are clipped to [1e-4, 1-1e-4] before the logit, as in the Lancet AN-0001-01 calibration analysis; this affects GPT-4o-mini log-probability scores and verbalised 0 or 1 values. '
                           'Bootstrap over reviews, B=1000, seed 20260928; resamples whose recalibration fit did not converge are dropped and counted.')
out['runtime_seconds'] = round(time.time() - T0, 1)
out['finished'] = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
import csv
with open('per_review_auc_common1999.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(per[0])); w.writeheader(); w.writerows(per)
json.dump(out, open('results.json', 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('n_common', 'macro_auc_point', 'two_stage', 'refused_sensitivity')}, indent=1))
for level in ('final', 'ta'):
    for m in models:
        r = cal[level][m]; print(level, m, {q: (round(r[q], 4) if isinstance(r[q], float) else r[q]) for q in ('n', 'O_E_ratio', 'recal_intercept', 'recal_slope', 'brier', 'ece_10_equal_width', 'n_scores_clipped_by_1e-4', 'n_distinct_scores')}, r['bootstrap_ci_reviews'])
print('runtime', out['runtime_seconds'])
