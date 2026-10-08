# AN-0001-06 (npj round 1): out-of-sample (out-of-bag) reliability of the tau selection procedure.
# Selection routine copied from Lancet AN-0001-05 tau_sensitivity.py (select_tau, matrices; grid of synergy/stop_eval.py).
# In each of 5000 bootstrap resamples of the 73 development reviews, tau is selected by the rule of the plan on the
# searched grid and applied to the development reviews not drawn in that resample (out-of-bag, OOB).
# Also: held-out and CLEF outcomes at tau=0.07 by review size (<= 2000, > 2000 records). Post hoc; no model call.
import json, datetime, time
import numpy as np, pandas as pd
ROOT = '/N/project/AiLab/jev'
SEED = 20260928; B = 5000; T0 = time.time()
TAUS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]
TAU0 = 0.07
def wilson(k, n, z=1.959964):
    if n == 0: return (float('nan'), float('nan'))
    ph = k / n; d = 1 + z * z / n; c = ph + z * z / (2 * n); h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)); return ((c - h) / d, (c + h) / d)
def load(recf, jevf, label_or0=False):
    rec = json.load(open(recf)); jev = {d['id']: d['p'] for d in json.load(open(jevf)) if d.get('ok') and d.get('p') is not None}
    return pd.DataFrame([(r['review'], int((r['label'] or 0) if label_or0 else r['label']), float(jev[r['id']])) for r in rec if r['id'] in jev], columns=['review', 'yy', 'p'])
def keep_pos(d):
    k = [r for r, g in d.groupby('review', sort=False) if g.yy.sum() > 0]; return d[d.review.isin(k)]
def matrices(d, taus):
    revs = list(dict.fromkeys(d.review)); REC = np.zeros((len(revs), len(taus))); WORK = np.zeros_like(REC); N = np.zeros(len(revs))
    for i, k in enumerate(revs):
        g = d[d.review == k]; p = g.p.values; y = g.yy.values; N[i] = len(g)
        for j, t in enumerate(taus):
            s = p >= t; WORK[i, j] = s.mean(); REC[i, j] = y[s].sum() / y.sum()
    return revs, REC, WORK, N
def select_tau(rel_vec, taus, target=0.95):
    ok = [t for t, r in zip(taus, rel_vec) if r >= target]; return max(ok) if ok else None
DEV = keep_pos(load(ROOT + '/synergy/dev2000_records.json', ROOT + '/synergy/jev_dev2000.json'))
revs, REC, WORK, N = matrices(DEV, TAUS); nrev = len(revs); RELB = REC >= 0.95
out = {'analysis_id': 'AN-0001-06', 'seed': SEED, 'B': B, 'grid': TAUS, 'n_dev_reviews': nrev,
       'selection_rule': 'largest tau of the grid at which at least 95% of the (resampled) development reviews reach recall >= 0.95',
       'full_sample_selection': select_tau(RELB.mean(0), TAUS)}
def stratum(n):
    return '<=500' if n <= 500 else ('501-1000' if n <= 1000 else '1001-2000')
strat = np.array([stratum(n) for n in N]); SNAMES = ['<=500', '501-1000', '1001-2000']
out['dev_reviews_per_stratum'] = {s: int((strat == s).sum()) for s in SNAMES}
rng = np.random.default_rng(SEED)
rows = []; sel_counts = {}
tot = {s: [0, 0] for s in SNAMES + ['all']}
for b in range(B):
    idx = rng.integers(0, nrev, nrev); t = select_tau(RELB[idx].mean(0), TAUS)
    sel_counts[t] = sel_counts.get(t, 0) + 1
    oob = np.setdiff1d(np.arange(nrev), np.unique(idx))
    if t is None or len(oob) == 0:
        rows.append({'b': b, 'tau': t, 'n_oob': int(len(oob))}); continue
    j = TAUS.index(t); r = {'b': b, 'tau': t, 'n_oob': int(len(oob)), 'oob_reliability': float(RELB[oob, j].mean()), 'oob_mean_recall': float(REC[oob, j].mean()),
                            'oob_min_recall': float(REC[oob, j].min()), 'oob_macro_work': float(WORK[oob, j].mean()), 'oob_n_reliable': int(RELB[oob, j].sum())}
    tot['all'][0] += int(RELB[oob, j].sum()); tot['all'][1] += len(oob)
    for s in SNAMES:
        o = oob[strat[oob] == s]
        if len(o):
            r[f'n_oob_{s}'] = int(len(o)); r[f'oob_reliability_{s}'] = float(RELB[o, j].mean()); r[f'oob_macro_work_{s}'] = float(WORK[o, j].mean()); r[f'oob_min_recall_{s}'] = float(REC[o, j].min())
            tot[s][0] += int(RELB[o, j].sum()); tot[s][1] += len(o)
    rows.append(r)
D = pd.DataFrame(rows); D.to_csv('bootstrap_oob.csv', index=False)
def summ(col):
    v = D[col].dropna().values
    return {'n_resamples': int(len(v)), 'mean': float(v.mean()), 'p2.5': float(np.percentile(v, 2.5)), 'p50': float(np.percentile(v, 50)), 'p97.5': float(np.percentile(v, 97.5)), 'min': float(v.min()),
            'proportion_of_resamples_below_0.90': float((v < 0.90).mean()), 'proportion_of_resamples_below_0.95': float((v < 0.95).mean())}
out['selected_tau_distribution'] = {str(k): v for k, v in sorted(sel_counts.items(), key=lambda x: (x[0] is None, x[0]))}
out['oob_overall'] = {'reliability': summ('oob_reliability'), 'mean_recall': summ('oob_mean_recall'), 'min_recall': summ('oob_min_recall'), 'macro_work': summ('oob_macro_work'), 'n_oob': summ('n_oob')}
out['oob_by_stratum'] = {s: {'reliability': summ(f'oob_reliability_{s}'), 'macro_work': summ(f'oob_macro_work_{s}'), 'min_recall': summ(f'oob_min_recall_{s}')} for s in SNAMES}
out['oob_pooled_over_resamples'] = {s: {'reliable': tot[s][0], 'oob_reviews': tot[s][1], 'proportion': tot[s][0] / tot[s][1] if tot[s][1] else None} for s in tot}
# held-out and CLEF at tau=0.07 by size
def by_size(d, name):
    res = {}
    g = [(k, x) for k, x in d.groupby('review', sort=False)]
    rows = []
    for k, x in g:
        s = x.p.values >= TAU0; y = x.yy.values
        rows.append({'review': k, 'N': len(x), 'recall': y[s].sum() / y.sum(), 'work': s.mean(), 'read': int(s.sum())})
    T = pd.DataFrame(rows)
    for lab, m in [('<=2000', T.N <= 2000), ('>2000', T.N > 2000), ('all', T.N > 0)]:
        t = T[m]; k = int((t.recall >= 0.95).sum()); n = len(t); lo, hi = wilson(k, n)
        res[lab] = {'n_reviews': n, 'n_reliable': k, 'reliability': k / n if n else None, 'wilson95': [lo, hi], 'macro_work': float(t.work.mean()) if n else None,
                    'pooled_work': float(t.read.sum() / t.N.sum()) if n else None, 'min_recall': float(t.recall.min()) if n else None, 'failing_reviews': t[t.recall < 0.95].review.tolist()}
    T.to_csv(f'per_review_tau007_{name}.csv', index=False)
    return res
TEST = keep_pos(load(ROOT + '/synergy/test_records.json', ROOT + '/synergy/jev_test.json'))
CLEF = keep_pos(load(ROOT + '/clef/clef_records.json', ROOT + '/synergy/jev_clef.json', label_or0=True))
out['heldout_tau0.07_by_size'] = by_size(TEST, 'heldout'); out['clef_tau0.07_by_size'] = by_size(CLEF, 'clef')
out['note'] = 'tau was tuned only on development reviews of at most 2000 records; OOB reviews are therefore also at most 2000 records. OOB estimates re-use the development reviews and are not independent of the data that fixed the grid and the rule.'
out['runtime_seconds'] = round(time.time() - T0, 1); out['finished'] = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
json.dump(out, open('results.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items()}, indent=1, default=float))
