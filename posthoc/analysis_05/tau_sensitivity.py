# AN-0001-05: tau grid, sensitivity of the label-free stopping rule to tau, uncertainty of the tau selection, per-review margins.
# Reads original outputs read-only; writes only into this folder. Environment: synergy/.venv.
import json, os, time, numpy as np, pandas as pd
ROOT = '/N/project/AiLab/jev'; OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 20260926; B = 5000; T0 = time.time()
TAUS_ORIG = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]              # grid in synergy/stop_eval.py (file time 2026-09-24 19:07 EDT)
TAUS_FINE = [round(0.01 * i, 2) for i in range(1, 11)] + [0.15, 0.20, 0.30, 0.40, 0.50]
TAU0 = 0.07
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)
def wilson(k, n, z=1.959964):
    if n == 0: return (float('nan'), float('nan'))
    ph = k / n; d = 1 + z * z / n; c = ph + z * z / (2 * n); h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)); return ((c - h) / d, (c + h) / d)
def load_synergy(split):
    rec = json.load(open(f'{ROOT}/synergy/{"dev2000" if split == "dev" else "test"}_records.json'))
    jev = {d['id']: d['p'] for d in json.load(open(f'{ROOT}/synergy/jev_{"dev2000" if split == "dev" else "test"}.json')) if d.get('ok') and d.get('p') is not None}
    return pd.DataFrame([(r['review'], int(r['label']), None if r['label_ta'] is None else int(r['label_ta']), float(jev[r['id']]), r['id']) for r in rec if r['id'] in jev], columns=['review', 'y', 'y_ta', 'p', 'id'])
def load_clef():
    rec = json.load(open(f'{ROOT}/clef/clef_records.json'))
    jev = {d['id']: d['p'] for d in json.load(open(f'{ROOT}/synergy/jev_clef.json')) if d.get('ok') and d.get('p') is not None}
    return pd.DataFrame([(r['review'], int(r['label'] or 0), None if r['label_ta'] is None else int(r['label_ta']), float(jev[r['id']]), r['id']) for r in rec if r['id'] in jev], columns=['review', 'y', 'y_ta', 'p', 'id'])
def select(df, level):
    col = 'y' if level == 'final' else 'y_ta'; keep = []
    for k, g in df.groupby('review', sort=False):
        if g[col].isna().any(): continue
        s = g[col].sum()
        if s > 0: keep.append(k)          # stop_eval.py / final_eval.py: reviews with at least one included study (final_eval also requires < N)
    d = df[df.review.isin(keep)].copy(); d['yy'] = d[col].astype(int); return d
def matrices(d, taus):
    revs = list(dict.fromkeys(d.review)); REC = np.zeros((len(revs), len(taus))); WORK = np.zeros_like(REC); N = np.zeros(len(revs)); NPOS = np.zeros(len(revs)); READ = np.zeros_like(REC); FOUND = np.zeros_like(REC)
    for i, k in enumerate(revs):
        g = d[d.review == k]; p = g.p.values; y = g.yy.values; N[i] = len(g); NPOS[i] = y.sum()
        for j, t in enumerate(taus):
            s = p >= t; READ[i, j] = s.sum(); FOUND[i, j] = y[s].sum(); WORK[i, j] = s.mean(); REC[i, j] = y[s].sum() / y.sum()
    return revs, REC, WORK, N, NPOS, READ, FOUND
def curve_table(name, d, taus):
    revs, REC, WORK, N, NPOS, READ, FOUND = matrices(d, taus); rows = []
    for j, t in enumerate(taus):
        ok = REC[:, j] >= 0.95; k = int(ok.sum()); n = len(revs); lo, hi = wilson(k, n)
        rows.append({'dataset': name, 'tau': t, 'n_reviews': n, 'n_reliable': k, 'reliability': k / n, 'wilson_lo': lo, 'wilson_hi': hi, 'n_fail': n - k, 'failing_reviews': [revs[i] for i in range(n) if not ok[i]],
                     'macro_work': float(WORK[:, j].mean()), 'pooled_work': float(READ[:, j].sum() / N.sum()), 'median_work': float(np.median(WORK[:, j])), 'mean_recall': float(REC[:, j].mean()), 'min_recall': float(REC[:, j].min()), 'pooled_recall': float(FOUND[:, j].sum() / NPOS.sum())})
    return rows, (revs, REC, WORK, N, NPOS)
def select_tau(rel_vec, taus, target=0.95):
    ok = [t for t, r in zip(taus, rel_vec) if r >= target]; return max(ok) if ok else None
DEV, TEST, CLEF = load_synergy('dev'), load_synergy('test'), load_clef(); log('loaded')
data = {'dev_final': select(DEV, 'final'), 'dev_ta': select(DEV, 'ta'), 'heldout_final': select(TEST, 'final'), 'heldout_ta': select(TEST, 'ta'), 'clef_final': select(CLEF, 'final'), 'clef_ta': select(CLEF, 'ta')}
results = {'analysis': 'AN-0001-05', 'seed': SEED, 'bootstrap_B': B, 'tau_grid_original': TAUS_ORIG, 'tau_grid_fine': TAUS_FINE, 'selection_rule': 'largest tau at which at least 95% of development reviews reach recall >= 0.95 (synergy/PREREG.md clarification of Sept 24, 19:00 to 19:02; freeze of Sept 25, 01:00)', 'curves': {}, 'mats': {}}
allrows = []; mats = {}
for name, d in data.items():
    rows, M = curve_table(name, d, TAUS_FINE); results['curves'][name] = rows; allrows += rows; mats[name] = M; log(name, 'reviews', len(M[0]))
pd.DataFrame([{k: v for k, v in r.items() if k != 'failing_reviews'} for r in allrows]).to_csv(f'{OUT}/tau_curves.csv', index=False)
# reproduction of synergy/stop_eval.py on the development reviews (original grid)
revs, REC, WORK, N, NPOS = mats['dev_final']; jo = [TAUS_FINE.index(t) for t in TAUS_ORIG]
rep = pd.DataFrame({'tau': TAUS_ORIG, 'reviews': len(revs), 'mean_work': WORK[:, jo].mean(0), 'mean_recall': REC[:, jo].mean(0), 'min_recall': REC[:, jo].min(0), 'reliab95': (REC[:, jo] >= 0.95).mean(0), 'reliab90': (REC[:, jo] >= 0.90).mean(0)})
rep.to_csv(f'{OUT}/stop_eval_reproduction_dev.csv', index=False); results['stop_eval_reproduction_dev'] = rep.round(4).to_dict('records')
results['dev_selection_on_original_grid'] = select_tau((REC[:, jo] >= 0.95).mean(0), TAUS_ORIG); results['dev_selection_on_fine_grid'] = select_tau((REC >= 0.95).mean(0), TAUS_FINE)
# reliability drop between 0.07 and 0.08 on development reviews (R3.1.04)
j7, j8 = TAUS_FINE.index(0.07), TAUS_FINE.index(0.08)
results['dev_reviews_losing_95_recall_between_0.07_and_0.08'] = [revs[i] for i in range(len(revs)) if REC[i, j7] >= 0.95 and REC[i, j8] < 0.95]
# implied held-out and CLEF outcomes for every tau (lookup)
def outcome_at(name, t):
    r = next(x for x in results['curves'][name] if x['tau'] == t); return {'reliability': r['reliability'], 'n_reliable': r['n_reliable'], 'n_reviews': r['n_reviews'], 'macro_work': r['macro_work'], 'pooled_work': r['pooled_work'], 'min_recall': r['min_recall']}
# bootstrap re-selection on development reviews
rng = np.random.default_rng(SEED); RELB = REC >= 0.95; nrev = len(revs)
def reselect(grid_idx, grid):
    sel = []
    for b in range(B):
        idx = rng.integers(0, nrev, nrev); rel = RELB[idx][:, grid_idx].mean(0); sel.append(select_tau(rel, grid))
    return sel
boot = {}
for gname, grid in [('original_grid', TAUS_ORIG), ('fine_grid', TAUS_FINE)]:
    rng = np.random.default_rng(SEED); gi = [TAUS_FINE.index(t) for t in grid]; sel = reselect(gi, grid)
    vals = [s for s in sel if s is not None]; cnt = pd.Series(sel, dtype=object).value_counts(dropna=False).to_dict()
    dist = []
    for t in grid:
        c = sum(1 for s in sel if s == t)
        if c: dist.append({'tau': t, 'count': c, 'proportion': c / B, 'heldout_final': outcome_at('heldout_final', t), 'clef_final': outcome_at('clef_final', t), 'heldout_ta': outcome_at('heldout_ta', t), 'clef_ta': outcome_at('clef_ta', t)})
    none = sum(1 for s in sel if s is None)
    imp_h = [outcome_at('heldout_final', s)['reliability'] for s in vals]; imp_c = [outcome_at('clef_final', s)['reliability'] for s in vals]
    boot[gname] = {'grid': grid, 'B': B, 'seed': SEED, 'n_no_tau_meets_rule': none, 'selected_tau_distribution': dist, 'tau_percentiles_2.5_50_97.5': [float(np.percentile(vals, q)) for q in (2.5, 50, 97.5)] if vals else None,
                   'proportion_selecting_0.07': sum(1 for s in sel if s == 0.07) / B, 'proportion_selecting_below_0.07': sum(1 for s in vals if s < 0.07) / B, 'proportion_selecting_above_0.07': sum(1 for s in vals if s > 0.07) / B,
                   'implied_heldout_final_reliability_mean': float(np.mean(imp_h)) if imp_h else None, 'implied_heldout_final_reliability_min': float(np.min(imp_h)) if imp_h else None,
                   'implied_clef_final_reliability_mean': float(np.mean(imp_c)) if imp_c else None, 'implied_clef_final_reliability_min': float(np.min(imp_c)) if imp_c else None, 'implied_clef_final_reliability_max': float(np.max(imp_c)) if imp_c else None}
results['bootstrap_reselection'] = boot
# leave-one-review-out re-selection
loo = {}
for gname, grid in [('original_grid', TAUS_ORIG), ('fine_grid', TAUS_FINE)]:
    gi = [TAUS_FINE.index(t) for t in grid]; sel = []
    for i in range(nrev):
        m = np.ones(nrev, bool); m[i] = False; sel.append(select_tau(RELB[m][:, gi].mean(0), grid))
    loo[gname] = {'selected': {revs[i]: sel[i] for i in range(nrev)}, 'distribution': {str(t): sum(1 for s in sel if s == t) for t in sorted(set(sel), key=lambda x: (x is None, x))}, 'all_equal_0.07': all(s == 0.07 for s in sel)}
results['leave_one_review_out'] = loo
# per-review margins at tau = 0.07
def margins(name):
    d = data[name]; rows = []
    for k, g in d.groupby('review', sort=False):
        p = g.p.values; y = g.yy.values; pi = np.sort(p[y == 1]); n1 = len(pi); m = int(np.floor(0.05 * n1 + 1e-9)); s = p >= TAU0
        rows.append({'dataset': name, 'review': k, 'N': len(g), 'n_included': n1, 'prevalence': n1 / len(g), 'min_included_p': float(pi[0]), 'n_included_below_0.07': int((pi < TAU0).sum()), 'n_included_0.07_to_0.09': int(((pi >= 0.07) & (pi < 0.09)).sum()),
                     'recall_at_0.07': float(y[s].sum() / n1), 'work_at_0.07': float(s.mean()), 'reliable_at_0.07': bool(y[s].sum() / n1 >= 0.95), 'max_tau_keeping_95_recall': float(pi[m]), 'max_grid_tau_keeping_95_recall': max([t for t in TAUS_FINE if y[p >= t].sum() / n1 >= 0.95], default=None)})
    return rows
mrows = []
for name in ['heldout_final', 'clef_final', 'heldout_ta', 'clef_ta', 'dev_final']: mrows += margins(name)
MR = pd.DataFrame(mrows); MR.to_csv(f'{OUT}/per_review_margins_tau0.07.csv', index=False)
results['margins_summary'] = {name: {'n_reviews': int((MR.dataset == name).sum()), 'n_with_min_included_p_below_0.07': int(((MR.dataset == name) & (MR.min_included_p < 0.07)).sum()), 'n_with_max_safe_tau_below_0.09': int(((MR.dataset == name) & (MR.max_tau_keeping_95_recall < 0.09)).sum()),
                                     'median_max_safe_tau': float(MR[MR.dataset == name].max_tau_keeping_95_recall.median()), 'min_max_safe_tau': float(MR[MR.dataset == name].max_tau_keeping_95_recall.min())} for name in MR.dataset.unique()}
results['runtime_seconds'] = round(time.time() - T0, 1)
json.dump(results, open(f'{OUT}/results.json', 'w'), indent=1, default=float); log('done', results['runtime_seconds'])
