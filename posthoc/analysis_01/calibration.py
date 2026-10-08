# AN-0001-01: post-hoc calibration assessment of Jev probabilities (round 1 revision).
# Reads original outputs read-only; writes only into this folder.
# Environment: synergy/.venv (Python 3.12.9, numpy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2).
import json, os, sys, time, warnings, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from scipy.optimize import brentq
ROOT = '/N/project/AiLab/jev'
OUT = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-4
SEED = 20260926
B_BOOT = 1000
rng = np.random.default_rng(SEED)
T0 = time.time()
def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)
def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS); return np.log(p / (1 - p))
def sigmoid(z): return 1 / (1 + np.exp(-z))
def wilson(k, n, z=1.959964):
    if n == 0: return (float('nan'), float('nan'))
    ph = k / n; d = 1 + z * z / n; c = ph + z * z / (2 * n); h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)
# ---------------- data loading ----------------
def load_synergy(split):
    rec = json.load(open(f'{ROOT}/synergy/{"dev2000" if split == "dev" else "test"}_records.json'))
    jev = {d['id']: d['p'] for d in json.load(open(f'{ROOT}/synergy/jev_{"dev2000" if split == "dev" else "test"}.json')) if d.get('ok') and d.get('p') is not None}
    rows = [(r['review'], int(r['label']), None if r['label_ta'] is None else int(r['label_ta']), float(jev[r['id']]), r['id']) for r in rec if r['id'] in jev]
    return pd.DataFrame(rows, columns=['review', 'y', 'y_ta', 'p', 'id'])
def load_clef():
    rec = json.load(open(f'{ROOT}/clef/clef_records.json'))
    jev = {d['id']: d['p'] for d in json.load(open(f'{ROOT}/synergy/jev_clef.json')) if d.get('ok') and d.get('p') is not None}
    rows = [(r['review'], int(r['label'] or 0), None if r['label_ta'] is None else int(r['label_ta']), float(jev[r['id']]), r['id']) for r in rec if r['id'] in jev]
    return pd.DataFrame(rows, columns=['review', 'y', 'y_ta', 'p', 'id'])
def select(df, level):
    """Reviews usable at a label level: all records labelled, 0 < positives < N."""
    col = 'y' if level == 'final' else 'y_ta'
    keep = []
    for k, g in df.groupby('review', sort=False):
        if g[col].isna().any(): continue
        s = g[col].sum()
        if 0 < s < len(g): keep.append(k)
    d = df[df.review.isin(keep)].copy(); d['yy'] = d[col].astype(int); return d
# ---------------- metrics ----------------
def fit_logit(x, y, X_extra=None):
    """Unpenalised logistic regression logit P(y=1) = a + b*x (+ extra columns). Two-parameter model: Newton-Raphson in numpy
    (maximum likelihood, tolerance 1e-10, at most 100 iterations; non-convergence or |b| > 50 flagged, e.g. complete separation).
    With extra columns (review fixed effects): scikit-learn LogisticRegression(penalty=None, lbfgs)."""
    if X_extra is None:
        X = np.column_stack([np.ones(len(x)), x]); beta = np.zeros(2); conv = False; it = 0
        for it in range(1, 101):
            eta = X @ beta; mu = sigmoid(eta); w = mu * (1 - mu); g = X.T @ (y - mu); H = (X * w[:, None]).T @ X
            try: step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError: break
            beta = beta + step
            if np.max(np.abs(step)) < 1e-10: conv = True; break
        if abs(beta[1]) > 50 or not np.all(np.isfinite(beta)): conv = False
        return float(beta[0]), float(beta[1]), conv, it
    X = np.column_stack([x, X_extra])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', ConvergenceWarning)
        m = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000, tol=1e-8).fit(X, y)
        conv = not any(issubclass(x_.category, ConvergenceWarning) for x_ in w)
    return float(m.intercept_[0]), float(m.coef_[0][0]), conv, int(np.ravel(m.n_iter_)[0])
def citl_intercept(lp, y):
    """Calibration-in-the-large: intercept a with slope fixed at 1 so that mean predicted equals observed rate."""
    prev = y.mean(); f = lambda a: sigmoid(lp + a).mean() - prev
    try: return float(brentq(f, -20, 20))
    except ValueError: return float('nan')
def ece_equal_width(p, y, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1); e = 0.0; rows = []
    for i in range(nb):
        m = b == i
        if m.sum() == 0: rows.append({'bin': i, 'lo': i / nb, 'hi': (i + 1) / nb, 'n': 0}); continue
        e += m.mean() * abs(y[m].mean() - p[m].mean())
        rows.append({'bin': i, 'lo': i / nb, 'hi': (i + 1) / nb, 'n': int(m.sum()), 'mean_p': float(p[m].mean()), 'obs': float(y[m].mean()), 'n_pos': int(y[m].sum())})
    return float(e), rows
def deciles(p, y, nb=10):
    o = np.argsort(p, kind='stable'); parts = np.array_split(o, nb); rows = []
    for i, idx in enumerate(parts):
        k = int(y[idx].sum()); n = len(idx); lo, hi = wilson(k, n)
        rows.append({'decile': i + 1, 'n': n, 'p_min': float(p[idx].min()), 'p_max': float(p[idx].max()), 'mean_p': float(p[idx].mean()), 'obs': k / n, 'n_pos': k, 'obs_lo': lo, 'obs_hi': hi})
    return rows
def core(p, y):
    lp = logit(p); prev = float(y.mean()); mp = float(p.mean())
    a, b, conv, nit = fit_logit(lp, y)
    brier = float(np.mean((p - y) ** 2)); bref = prev * (1 - prev)
    ece, _ = ece_equal_width(p, y)
    return {'n': int(len(y)), 'n_pos': int(y.sum()), 'observed_rate': prev, 'mean_predicted': mp, 'O_E_ratio': float(y.sum() / p.sum()),
            'citl_intercept_slope1': citl_intercept(lp, y), 'recal_intercept': a, 'recal_slope': b, 'recal_converged': conv, 'recal_iter': nit,
            'brier': brier, 'brier_reference': float(bref), 'scaled_brier': float(1 - brier / bref) if bref > 0 else float('nan'), 'ece_10_equal_width': ece}
def analyse(name, d, per_review=True, boot=True):
    """d: DataFrame with columns review, yy, p."""
    log('analyse', name, len(d), 'records', d.review.nunique(), 'reviews')
    p = d.p.values.astype(float); y = d.yy.values.astype(int); res = {'dataset': name, 'n_reviews': int(d.review.nunique())}
    res.update(core(p, y))
    res['pooled_auc'] = float(roc_auc_score(y, p))
    # review fixed effects for the intercept (dummies, reference = first review), slope shared
    if per_review and d.review.nunique() > 1:
        revs = sorted(d.review.unique()); codes = pd.Categorical(d.review, categories=revs).codes
        D = np.zeros((len(d), len(revs) - 1)); m = codes > 0; D[np.arange(len(d))[m], codes[m] - 1] = 1
        a_fe, b_fe, conv_fe, nit_fe = fit_logit(logit(p), y, D)
        res['recal_slope_review_fixed_effects'] = b_fe; res['recal_fe_converged'] = conv_fe
    res['deciles'] = deciles(p, y); res['ece_bins'] = ece_equal_width(p, y)[1]
    # per review
    if per_review:
        rows = []
        for k, g in d.groupby('review', sort=False):
            pp = g.p.values.astype(float); yy = g.yy.values.astype(int); lp = logit(pp)
            row = {'review': k, 'n': len(g), 'n_pos': int(yy.sum()), 'observed_rate': float(yy.mean()), 'mean_predicted': float(pp.mean()), 'O_E_ratio': float(yy.sum() / pp.sum()),
                   'citl_intercept_slope1': citl_intercept(lp, yy), 'brier': float(np.mean((pp - yy) ** 2)), 'ece_10_equal_width': ece_equal_width(pp, yy)[0], 'auc': float(roc_auc_score(yy, pp))}
            if yy.sum() >= 5:
                a, b, conv, nit = fit_logit(lp, yy); row.update({'recal_intercept': a, 'recal_slope': b, 'recal_converged': conv, 'recal_iter': nit})
            rows.append(row)
        PR = pd.DataFrame(rows); PR.to_csv(f'{OUT}/per_review_{name}.csv', index=False)
        fit = PR[PR.n_pos >= 5]; ok = fit[fit.recal_converged == True] if 'recal_converged' in fit else fit.iloc[0:0]
        res['per_review'] = {'n_reviews_all': int(len(PR)), 'n_reviews_ge5_included': int(len(fit)), 'n_fits_converged': int(len(ok)), 'n_fits_not_converged': int(len(fit) - len(ok)),
                             'O_E_median': float(PR.O_E_ratio.median()), 'O_E_iqr': [float(PR.O_E_ratio.quantile(.25)), float(PR.O_E_ratio.quantile(.75))], 'O_E_range': [float(PR.O_E_ratio.min()), float(PR.O_E_ratio.max())],
                             'n_reviews_O_E_within_0.8_1.25': int(((PR.O_E_ratio >= 0.8) & (PR.O_E_ratio <= 1.25)).sum()),
                             'intercept_median': float(ok.recal_intercept.median()) if len(ok) else None, 'intercept_iqr': [float(ok.recal_intercept.quantile(.25)), float(ok.recal_intercept.quantile(.75))] if len(ok) else None,
                             'slope_median': float(ok.recal_slope.median()) if len(ok) else None, 'slope_iqr': [float(ok.recal_slope.quantile(.25)), float(ok.recal_slope.quantile(.75))] if len(ok) else None,
                             'slope_range': [float(ok.recal_slope.min()), float(ok.recal_slope.max())] if len(ok) else None,
                             'n_slopes_within_0.8_1.2': int(((ok.recal_slope >= 0.8) & (ok.recal_slope <= 1.2)).sum()) if len(ok) else None,
                             'ece_median': float(PR.ece_10_equal_width.median()), 'brier_median': float(PR.brier.median()), 'macro_auc': float(PR.auc.mean())}
    # bootstrap over reviews for pooled quantities
    if boot and d.review.nunique() > 1:
        groups = {k: (g.p.values.astype(float), g.yy.values.astype(int)) for k, g in d.groupby('review', sort=False)}; keys = list(groups); B = {'O_E_ratio': [], 'recal_intercept': [], 'recal_slope': [], 'ece_10_equal_width': [], 'brier': [], 'citl_intercept_slope1': []}
        r2 = np.random.default_rng(SEED + 1)
        for b_ in range(B_BOOT):
            idx = r2.integers(0, len(keys), len(keys)); pp = np.concatenate([groups[keys[i]][0] for i in idx]); yy = np.concatenate([groups[keys[i]][1] for i in idx])
            if yy.sum() == 0 or yy.sum() == len(yy): continue
            lp = logit(pp); a, bb, conv, nit = fit_logit(lp, yy)
            B['O_E_ratio'].append(yy.sum() / pp.sum()); B['recal_intercept'].append(a); B['recal_slope'].append(bb); B['ece_10_equal_width'].append(ece_equal_width(pp, yy)[0]); B['brier'].append(np.mean((pp - yy) ** 2)); B['citl_intercept_slope1'].append(citl_intercept(lp, yy))
        res['bootstrap_ci_reviews'] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in B.items()}; res['bootstrap_ci_reviews']['B'] = len(B['recal_slope']); res['bootstrap_ci_reviews']['seed'] = SEED + 1
    return res
results = {'analysis': 'AN-0001-01', 'seed': SEED, 'eps_clip': EPS, 'datasets': {}}
# ---------------- SYNERGY and CLEF ----------------
DEV = load_synergy('dev'); TEST = load_synergy('test'); CLEF = load_clef()
log('loaded dev', len(DEV), 'test', len(TEST), 'clef', len(CLEF))
for name, df, level in [('dev_final_batched', DEV, 'final'), ('dev_ta_batched', DEV, 'ta'), ('heldout_final_batched', TEST, 'final'), ('heldout_ta_batched', TEST, 'ta'), ('clef_final_batched', CLEF, 'final'), ('clef_ta_batched', CLEF, 'ta')]:
    results['datasets'][name] = analyse(name, select(df, level))
# same 12 held-out reviews at both label levels (for a like-for-like comparison)
ta12 = select(TEST, 'ta').review.unique(); d12 = select(TEST, 'final'); d12 = d12[d12.review.isin(ta12)]
results['datasets']['heldout12_final_batched'] = analyse('heldout12_final_batched', d12)
# ---------------- 2002-record subset: single vs batched ----------------
ids = set(json.load(open(f'{ROOT}/synergy/test_cmp_ids.json')))
L = lambda f: {d['id']: d['p'] for d in json.load(open(f'{ROOT}/synergy/{f}')) if d.get('ok') and d.get('p') is not None}
S1, B10sub, B10full = L('jev_testcmp_single.json'), L('jev_testcmp_b10.json'), L('jev_test.json')
sub = TEST[TEST.id.isin(ids)].copy(); sub = sub[sub.id.isin(S1) & sub.id.isin(B10sub)]
results['subset_common_records'] = int(len(sub))
for mode, M in [('single', S1), ('b10_subset', B10sub), ('b10_full', B10full)]:
    s = sub.copy(); s['p'] = s.id.map(M).astype(float)
    results['datasets'][f'subset2002_final_{mode}'] = analyse(f'subset2002_final_{mode}', select(s, 'final'))
    st = select(s, 'ta')
    if len(st): results['datasets'][f'subset2002_ta_{mode}'] = analyse(f'subset2002_ta_{mode}', st)
# paired per-review comparison single vs b10_full (slope, intercept, O:E, ECE, Brier)
try:
    A = pd.read_csv(f'{OUT}/per_review_subset2002_final_single.csv').set_index('review'); Bf = pd.read_csv(f'{OUT}/per_review_subset2002_final_b10_full.csv').set_index('review')
    comp = {}
    conv = (A.get('recal_converged') == True) & (Bf.get('recal_converged') == True)   # coefficient comparisons restricted to reviews whose fits converged in both modes
    for c in ['mean_predicted', 'O_E_ratio', 'ece_10_equal_width', 'brier', 'recal_slope', 'recal_intercept']:
        if c in A and c in Bf:
            x = A[c]; z = Bf[c]; ok = x.notna() & z.notna()
            if c.startswith('recal_'): ok = ok & conv
            comp[c] = {'n': int(ok.sum()), 'mean_single': float(x[ok].mean()), 'mean_b10_full': float(z[ok].mean()), 'median_single': float(x[ok].median()), 'median_b10_full': float(z[ok].median()), 'mean_diff_b10_minus_single': float((z - x)[ok].mean()), 'n_b10_higher': int(((z - x)[ok] > 0).sum())}
    results['subset2002_per_review_batched_minus_single'] = comp
except Exception as e: results['subset2002_per_review_batched_minus_single'] = f'error: {e}'
# ---------------- pilot Alzheimer's disease review (aggregated only) ----------------
DR = {r['id']: r for r in json.load(open(f'{ROOT}/synergy/dhl_records.json'))}
Bt = [d for d in json.load(open(f'{ROOT}/synergy/dhl_jev_b10.json')) if d.get('ok')]
Sg = [d for d in json.load(open(f'{ROOT}/screen/screen_jev.json')) if d.get('ok')]
def pilot_df(items, key):
    rows = []
    for d in items:
        r = DR[d[key]]; rows.append(('pilot', 1 if r['stage'] == 'included' else 0, 1 if r['stage'] in ('included', 'fulltext_excluded') else 0, float(d['p']), d[key]))
    return pd.DataFrame(rows, columns=['review', 'y', 'y_ta', 'p', 'id'])
PB = pilot_df(Bt, 'id'); PS = pilot_df(Sg, 'pmid')
assert PB.y_ta.sum() == 144 and PB.y.sum() == 77 and PS.y_ta.sum() == 144 and PS.y.sum() == 77, (PB.y_ta.sum(), PB.y.sum(), PS.y_ta.sum(), PS.y.sum())
for name, df, level in [('pilot_ta_batched', PB, 'ta'), ('pilot_final_batched', PB, 'final'), ('pilot_ta_single', PS, 'ta'), ('pilot_final_single', PS, 'final')]:
    results['datasets'][name] = analyse(name, select(df, level), per_review=False, boot=False)
# ---------------- pre-committed descriptor rule ----------------
rule = {}
for level in ['final', 'ta']:
    h = results['datasets'][f'heldout_{level}_batched']; c = results['datasets'][f'clef_{level}_batched']
    ok_slope = all(0.8 <= x['recal_slope'] <= 1.2 for x in (h, c)); ok_oe = all(0.8 <= x['O_E_ratio'] <= 1.25 for x in (h, c))
    rule[level] = {'heldout_slope': h['recal_slope'], 'clef_slope': c['recal_slope'], 'heldout_O_E': h['O_E_ratio'], 'clef_O_E': c['O_E_ratio'], 'slope_in_0.8_1.2_both': ok_slope, 'O_E_in_0.8_1.25_both': ok_oe, 'rule_met': bool(ok_slope and ok_oe)}
results['descriptor_rule'] = {'rule': 'The word calibrated may be used (post hoc, qualified) only if the pooled recalibration slope lies in 0.8 to 1.2 and the O:E ratio in 0.8 to 1.25 for that label level in both the held-out and the CLEF sets (REVISION_ANALYSIS_LOG.md, AN-0001-01, 2026-09-26 17:43:28 EDT).', 'by_label_level': rule, 'any_level_met': bool(any(v['rule_met'] for v in rule.values()))}
results['runtime_seconds'] = round(time.time() - T0, 1)
json.dump(results, open(f'{OUT}/results.json', 'w'), indent=1, default=float)
# summary table
rows = []
for k, v in results['datasets'].items():
    rows.append({'dataset': k, 'n_reviews': v.get('n_reviews'), 'n_records': v['n'], 'n_pos': v['n_pos'], 'observed_rate': v['observed_rate'], 'mean_predicted': v['mean_predicted'], 'O_E_ratio': v['O_E_ratio'], 'citl_intercept_slope1': v['citl_intercept_slope1'],
                 'recal_intercept': v['recal_intercept'], 'recal_slope': v['recal_slope'], 'recal_slope_fe': v.get('recal_slope_review_fixed_effects'), 'brier': v['brier'], 'scaled_brier': v['scaled_brier'], 'ece': v['ece_10_equal_width'], 'pooled_auc': v['pooled_auc'],
                 'slope_ci': v.get('bootstrap_ci_reviews', {}).get('recal_slope'), 'O_E_ci': v.get('bootstrap_ci_reviews', {}).get('O_E_ratio'), 'per_review_slope_median': (v.get('per_review') or {}).get('slope_median'), 'per_review_O_E_median': (v.get('per_review') or {}).get('O_E_median')})
pd.DataFrame(rows).to_csv(f'{OUT}/summary_table.csv', index=False)
log('done in', results['runtime_seconds'], 's')
