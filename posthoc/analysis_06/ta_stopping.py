# AN-0001-06: label-free threshold rule (tau = 0.07) against title and abstract labels; pilot review records below tau.
# Reads original outputs read-only; writes only into this folder. Environment: synergy/.venv.
import json, os, time, numpy as np, pandas as pd
ROOT = '/N/project/AiLab/jev'; OUT = os.path.dirname(os.path.abspath(__file__)); TAU0 = 0.07; T0 = time.time()
def wilson(k, n, z=1.959964):
    ph = k / n; d = 1 + z * z / n; c = ph + z * z / (2 * n); h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)); return ((c - h) / d, (c + h) / d)
def load_synergy(split):
    rec = json.load(open(f'{ROOT}/synergy/{"dev2000" if split == "dev" else "test"}_records.json'))
    jev = {d['id']: d['p'] for d in json.load(open(f'{ROOT}/synergy/jev_{"dev2000" if split == "dev" else "test"}.json')) if d.get('ok') and d.get('p') is not None}
    return pd.DataFrame([(r['review'], int(r['label']), None if r['label_ta'] is None else int(r['label_ta']), float(jev[r['id']]), r['id']) for r in rec if r['id'] in jev], columns=['review', 'y', 'y_ta', 'p', 'id'])
def load_clef():
    rec = json.load(open(f'{ROOT}/clef/clef_records.json'))
    jev = {d['id']: d['p'] for d in json.load(open(f'{ROOT}/synergy/jev_clef.json')) if d.get('ok') and d.get('p') is not None}
    return pd.DataFrame([(r['review'], int(r['label'] or 0), None if r['label_ta'] is None else int(r['label_ta']), float(jev[r['id']]), r['id']) for r in rec if r['id'] in jev], columns=['review', 'y', 'y_ta', 'p', 'id'])
def evaluate(name, d, col, taus=(TAU0,)):
    keep = [k for k, g in d.groupby('review', sort=False) if not g[col].isna().any() and g[col].sum() > 0]; d = d[d.review.isin(keep)]
    out = {}
    for t in taus:
        rows = []
        for k, g in d.groupby('review', sort=False):
            p = g.p.values; y = g[col].astype(int).values; s = p >= t
            rows.append({'dataset': name, 'label': col, 'tau': t, 'review': k, 'N': len(g), 'n_positive': int(y.sum()), 'n_read': int(s.sum()), 'work': float(s.mean()), 'n_found': int(y[s].sum()), 'n_positive_below_tau': int(y[~s].sum()), 'recall': float(y[s].sum() / y.sum()), 'min_positive_p': float(p[y == 1].min())})
        R = pd.DataFrame(rows); k = int((R.recall >= 0.95).sum()); n = len(R); lo, hi = wilson(k, n)
        out[str(t)] = {'n_reviews': n, 'n_reliable': k, 'reliability': k / n, 'wilson_ci': [lo, hi], 'failing_reviews': R[R.recall < 0.95].review.tolist(), 'mean_recall': float(R.recall.mean()), 'min_recall': float(R.recall.min()), 'pooled_recall': float(R.n_found.sum() / R.n_positive.sum()),
                       'macro_work': float(R.work.mean()), 'pooled_work': float(R.n_read.sum() / R.N.sum()), 'median_work': float(R.work.median()), 'n_positive_total': int(R.n_positive.sum()), 'n_positive_below_tau_total': int(R.n_positive_below_tau.sum()), 'per_review': R}
    return out
TEST, CLEF = load_synergy('test'), load_clef()
ta12 = [k for k, g in TEST.groupby('review', sort=False) if not g.y_ta.isna().any() and g.y_ta.sum() > 0]
results = {'analysis': 'AN-0001-06', 'tau': TAU0, 'collections': {}}; tables = []
TAUS = (0.03, 0.05, 0.07, 0.10, 0.15, 0.20)
for name, d, col in [('heldout12_ta', TEST[TEST.review.isin(ta12)], 'y_ta'), ('heldout12_final', TEST[TEST.review.isin(ta12)], 'y'), ('heldout23_final', TEST, 'y'), ('clef31_ta', CLEF, 'y_ta'), ('clef28_final', CLEF, 'y')]:
    ev = evaluate(name, d, col, TAUS); results['collections'][name] = {}
    for t, v in ev.items():
        tables.append(v.pop('per_review')); results['collections'][name][t] = v
pd.concat(tables).to_csv(f'{OUT}/per_review_tau_ta.csv', index=False)
# pilot Alzheimer's disease review: aggregated counts only (I-14)
DR = {r['id']: r for r in json.load(open(f'{ROOT}/synergy/dhl_records.json'))}
Bt = {d['id']: d['p'] for d in json.load(open(f'{ROOT}/synergy/dhl_jev_b10.json')) if d.get('ok')}
Sg = {d['pmid']: d['p'] for d in json.load(open(f'{ROOT}/screen/screen_jev.json')) if d.get('ok')}
pilot = {'n_records': len(DR), 'n_passed_title_abstract': sum(1 for r in DR.values() if r['stage'] in ('included', 'fulltext_excluded')), 'n_final_included': sum(1 for r in DR.values() if r['stage'] == 'included'), 'n_fulltext_excluded': sum(1 for r in DR.values() if r['stage'] == 'fulltext_excluded'), 'modes': {}}
for mode, P in [('batched_10', Bt), ('single', Sg)]:
    m = {}
    for t in TAUS:
        below = lambda st: sum(1 for i, r in DR.items() if r['stage'] in st and P[i] < t)
        m[str(t)] = {'n_read': sum(1 for i in DR if P[i] >= t), 'work': sum(1 for i in DR if P[i] >= t) / len(DR), 'passed_ta_below_tau': below(('included', 'fulltext_excluded')), 'final_included_below_tau': below(('included',)), 'fulltext_excluded_below_tau': below(('fulltext_excluded',)),
                     'recall_ta': 1 - below(('included', 'fulltext_excluded')) / pilot['n_passed_title_abstract'], 'recall_final': 1 - below(('included',)) / pilot['n_final_included']}
    m['min_p_passed_ta'] = min(P[i] for i, r in DR.items() if r['stage'] in ('included', 'fulltext_excluded')); m['min_p_final_included'] = min(P[i] for i, r in DR.items() if r['stage'] == 'included')
    pilot['modes'][mode] = m
results['pilot_review'] = pilot; results['runtime_seconds'] = round(time.time() - T0, 1)
json.dump(results, open(f'{OUT}/results.json', 'w'), indent=1, default=float); print(json.dumps({k: results['collections'][k]['0.07'] for k in results['collections']}, indent=1, default=float)); print(json.dumps(pilot, indent=1, default=float))
