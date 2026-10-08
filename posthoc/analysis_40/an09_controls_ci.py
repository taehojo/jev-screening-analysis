# AN-0001-09 (npj round 1): bootstrap CIs for the two negative controls and the two rephrasings (12 development reviews).
# Adapted from synergy/robust_eval.py (same inputs and per-review AUC and Spearman definitions); adds exact AUCs,
# paired differences with review-level percentile bootstrap CIs, and counts. Post hoc. No model call.
import json, datetime
from fractions import Fraction as F
import numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score
S = '/N/project/AiLab/jev/synergy/'
SEED = 20260928; B = 5000
R = json.load(open(S + 'dev2000_records.json')); ids = set(json.load(open(S + 'robust_ids.json')))
lab = {r['id']: int(r['label']) for r in R if r['id'] in ids}; rev = {r['id']: r['review'] for r in R if r['id'] in ids}
load = lambda f: {d['id']: d['p'] for d in json.load(open(S + f)) if d.get('ok') and d.get('p') is not None}
Q0 = load('jev_dev2000.json')
conds = [('q1', 'jev_robust_q1.json'), ('q2', 'jev_robust_q2.json'), ('title_only', 'jev_robust_titleonly.json'), ('mismatch', 'jev_robust_mismatch.json')]
V = {c: load(f) for c, f in conds}
revs = sorted(set(rev.values()))
def auc_exact(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float); n1 = int(y.sum()); n0 = len(y) - n1
    r2 = np.rint(2 * rankdata(s, method='average')).astype(np.int64); u2 = int(r2[y == 1].sum()) - n1 * (n1 + 1)
    return F(u2, 2 * n1 * n0)
per = []
for k in revs:
    ii = sorted(i for i in ids if rev[i] == k); y = [lab[i] for i in ii]; a0 = [Q0[i] for i in ii]
    row = {'review': k, 'n': len(ii), 'n_pos': sum(y), 'auc_q0_exact': auc_exact(y, a0)}
    assert abs(float(row['auc_q0_exact']) - roc_auc_score(y, a0)) < 1e-12
    for c, _ in conds:
        assert all(i in V[c] for i in ii), (c, k)
        a1 = [V[c][i] for i in ii]
        row[f'auc_{c}_exact'] = auc_exact(y, a1); row[f'rho_{c}'] = float(spearmanr(a0, a1).correlation)
        assert abs(float(row[f'auc_{c}_exact']) - roc_auc_score(y, a1)) < 1e-12
    per.append(row)
n = len(per)
rng = np.random.default_rng(SEED); IDX = rng.integers(0, n, (B, n))
def boot(v):
    v = np.asarray(v, float); m = v[IDX].mean(1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
out = {'analysis_id': 'AN-0001-09', 'n_reviews': n, 'n_records': len(ids), 'bootstrap': {'B': B, 'seed': SEED, 'type': 'percentile, reviews resampled with replacement; one index matrix shared by all quantities'},
       'reviews': revs}
q0 = [float(r['auc_q0_exact']) for r in per]
out['original_q0'] = {'macro_auc': float(np.mean(q0)), 'ci': boot(q0)}
for c, _ in conds:
    a = [float(r[f'auc_{c}_exact']) for r in per]; d = [r[f'auc_{c}_exact'] - r['auc_q0_exact'] for r in per]; df = [float(x) for x in d]
    rho = [r[f'rho_{c}'] for r in per]
    out[c] = {'macro_auc': float(np.mean(a)), 'macro_auc_ci': boot(a), 'diff_from_original': float(np.mean(df)), 'diff_ci': boot(df),
              'reviews_lower_than_original': sum(1 for x in d if x < 0), 'reviews_higher': sum(1 for x in d if x > 0), 'reviews_equal': sum(1 for x in d if x == 0),
              'min_auc': float(min(a)), 'mean_rho': float(np.mean(rho)), 'mean_rho_ci': boot(rho), 'min_rho': float(min(rho))}
stored = json.load(open(S + 'robust_results.json'))
chk = {}
for c, _ in conds:
    s = stored[c]
    chk[c] = {'stored_auc_alt': s['auc_alt'], 'recomputed_rounded4': round(out[c]['macro_auc'], 4), 'stored_auc_q0': s['auc_q0'], 'recomputed_q0_rounded4': round(out['original_q0']['macro_auc'], 4),
              'stored_mean_rho': s['mean_rho'], 'recomputed_rho_rounded3': round(out[c]['mean_rho'], 3), 'stored_auc_diff': s['auc_diff'], 'recomputed_diff_rounded4': round(out[c]['diff_from_original'], 4)}
    chk[c]['all_match'] = (chk[c]['stored_auc_alt'] == chk[c]['recomputed_rounded4'] and chk[c]['stored_auc_q0'] == chk[c]['recomputed_q0_rounded4'] and chk[c]['stored_mean_rho'] == chk[c]['recomputed_rho_rounded3'] and chk[c]['stored_auc_diff'] == chk[c]['recomputed_diff_rounded4'])
out['reproduction_of_robust_results_json'] = chk
out['finished'] = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
import csv
with open('per_review.csv', 'w', newline='') as f:
    rows = [{k: (float(v) if isinstance(v, F) else v) for k, v in r.items()} for r in per]
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(out, open('results.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ('reviews',)}, indent=1))
