#!/usr/bin/env python
# AN-0001-04: uncertainty (Wilson, Clopper-Pearson, bootstrap) and estimands (macro, pooled, median, IQR, range) for every
# stopping-rule outcome; small-review restriction; strata by size and prevalence; screening band below tau; per-review tables.
# Reads original outputs and AN-0001-02 outputs only (read-only); writes into this folder only.
import os, sys, json, math, time, platform, datetime
import numpy as np, pandas as pd, scipy, sklearn
from scipy.stats import beta
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = '/N/project/AiLab/jev'; SYN = ROOT + '/synergy'; CLEF = ROOT + '/clef'
AN02 = os.path.join(os.path.dirname(HERE), 'AN-0001-02', 'per_review.csv')
TAU = 0.07; B = 5000; BOOT_SEED = 1004; Z = 1.959963984540054
t0 = time.time(); started = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
def log(*a): print(*a, flush=True)

def wilson(k, n):
    p = k / n; den = 1 + Z * Z / n; c = (p + Z * Z / (2 * n)) / den; h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return [c - h, c + h]
def clopper(k, n):
    lo = float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0; hi = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    return [lo, hi]
def prop(k, n):
    return {'k': int(k), 'n': int(n), 'proportion': k / n, 'wilson95': wilson(k, n), 'clopper_pearson95': clopper(k, n)}
def boot_mean(v, seed=BOOT_SEED, Bn=B):
    v = np.asarray(v, float); rng = np.random.default_rng(seed); m = v[rng.integers(0, len(v), (Bn, len(v)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
def dist(v):
    v = np.asarray(v, float)
    return {'mean': float(v.mean()), 'boot95_of_mean': boot_mean(v), 'median': float(np.median(v)), 'q1': float(np.percentile(v, 25)), 'q3': float(np.percentile(v, 75)), 'min': float(v.min()), 'max': float(v.max()), 'n': int(len(v))}
def pooled(work, N):
    return float((np.asarray(work) * np.asarray(N)).sum() / np.asarray(N).sum())

def load_by(path):
    by = {}
    for r in json.load(open(path)): by.setdefault(r['review'], []).append(r)
    return by
def load_jev(path): return {d['id']: d['p'] for d in json.load(open(path)) if d.get('ok') and d.get('p') is not None}

def threshold_rows(by, J, label='label'):
    rows = []
    for k, rs in by.items():
        if not all(r['id'] in J for r in rs): continue
        if any(r.get(label) is None for r in rs): continue
        y = np.array([int(r[label]) for r in rs]); p = np.array([float(J[r['id']]) for r in rs]); N = len(y); n1 = int(y.sum())
        if not (0 < n1 < N): continue
        s = p >= TAU; pin = p[y == 1]
        b1 = (p >= 0.03) & (p < 0.07); b2 = (p >= 0.01) & (p < 0.07)
        rows.append({'review': k, 'N': N, 'n1': n1, 'prevalence': n1 / N, 'jev_auc': float(roc_auc_score(y, p)),
                     'thr_work': float(s.mean()), 'thr_read_n': int(s.sum()), 'thr_rec': float(y[s].sum() / n1), 'thr_found_n': int(y[s].sum()), 'thr_reliable': int(y[s].sum() / n1 >= 0.95),
                     'min_included_p': float(pin.min()), 'n_included_below_tau': int((pin < TAU).sum()),
                     'band_0.03_0.07_records_frac': float(b1.mean()), 'band_0.01_0.07_records_frac': float(b2.mean()), 'band_0.03_0.07_records_n': int(b1.sum()), 'band_0.01_0.07_records_n': int(b2.sum()),
                     'band_0.03_0.07_included_frac': float(b1[y == 1].mean()), 'band_0.01_0.07_included_frac': float(b2[y == 1].mean()), 'band_0.03_0.07_included_n': int(b1[y == 1].sum()), 'band_0.01_0.07_included_n': int(b2[y == 1].sum())})
    return pd.DataFrame(rows).set_index('review')

def al_columns(path, methods):
    A = pd.DataFrame([d for d in json.load(open(path)) if d.get('wss95') is not None])
    A['knee_rel_seed'] = (A.knee_rec >= 0.95).astype(float)
    out = {}
    for m, tag in methods:
        S = A[A.method == m]
        if len(S) == 0: continue
        g = S.groupby('review')[['wss95', 'wss100', 'knee_work', 'knee_rec', 'knee_rel_seed']].mean()
        out[f'{tag}_wss95'] = g.wss95; out[f'{tag}_wss100'] = g.wss100; out[f'{tag}_knee_work'] = g.knee_work; out[f'{tag}_knee_rec'] = g.knee_rec
        out[f'{tag}_knee_reliable'] = (g.knee_rec >= 0.95).astype(int); out[f'{tag}_knee_rel_perseed'] = g.knee_rel_seed; out[f'{tag}_n_seeds'] = S.groupby('review').seed.nunique()
    return pd.DataFrame(out)

def rule_summary(T, work_col, rec_col, rel_col, name, perseed_col=None):
    k = int(T[rel_col].sum()); n = int(len(T)); fails = T[T[rel_col] == 0]
    s = {'rule': name, 'reliability': prop(k, n), 'work_macro': dist(T[work_col]), 'work_pooled_record_weighted': pooled(T[work_col], T.N),
         'recall_macro': dist(T[rec_col]), 'failing_reviews': [{'review': i, 'recall': float(r[rec_col]), 'work': float(r[work_col]), 'N': int(r.N), 'n1': int(r.n1)} for i, r in fails.iterrows()]}
    if rec_col == 'thr_rec': s['recall_pooled'] = float(T.thr_found_n.sum() / T.n1.sum())
    if perseed_col is not None: s['reliability_per_seed_mean'] = float(T[perseed_col].mean())
    return s

def strata(T, knee_tag):
    rows = []
    T = T.copy(); T['size_stratum'] = np.where(T.N <= 2000, 'N<=2000', 'N>2000')
    try: T['prev_tertile'] = pd.qcut(T.prevalence, 3, labels=['low', 'mid', 'high']).astype(str)
    except Exception: T['prev_tertile'] = 'n/a'
    cuts = [float(x) for x in np.percentile(T.prevalence, [100 / 3, 200 / 3])]
    for var in ['size_stratum', 'prev_tertile']:
        for lev, S in T.groupby(var):
            r = {'stratum_variable': var, 'stratum': lev, 'n_reviews': int(len(S)), 'records': int(S.N.sum()), 'prevalence_min': float(S.prevalence.min()), 'prevalence_max': float(S.prevalence.max()),
                 'thr_reliable_k': int(S.thr_reliable.sum()), 'thr_reliability': float(S.thr_reliable.mean()), 'thr_wilson_lo': wilson(int(S.thr_reliable.sum()), len(S))[0], 'thr_wilson_hi': wilson(int(S.thr_reliable.sum()), len(S))[1],
                 'thr_work_macro': float(S.thr_work.mean()), 'thr_work_pooled': pooled(S.thr_work, S.N), 'thr_recall_macro': float(S.thr_rec.mean()), 'thr_recall_min': float(S.thr_rec.min())}
            kc = f'{knee_tag}_knee_reliable'
            if kc in S:
                kk = int(S[kc].sum()); r.update({'knee_reliable_k': kk, 'knee_reliability': kk / len(S), 'knee_wilson_lo': wilson(kk, len(S))[0], 'knee_wilson_hi': wilson(kk, len(S))[1], 'knee_work_macro': float(S[f'{knee_tag}_knee_work'].mean()), 'knee_work_pooled': pooled(S[f'{knee_tag}_knee_work'], S.N)})
            rows.append(r)
    return rows, cuts

def size_restriction(T, knee_tags, cut):
    out = {}
    for lab, S in [(f'N<{cut}', T[T.N < cut]), (f'N>={cut}', T[T.N >= cut])]:
        d = {'n_reviews': int(len(S)), 'reviews': S.index.tolist() if len(S) <= 10 else None, 'thr_work_macro': float(S.thr_work.mean()) if len(S) else None, 'thr_work_pooled': pooled(S.thr_work, S.N) if len(S) else None,
             'thr_reliable': prop(int(S.thr_reliable.sum()), len(S)) if len(S) else None}
        for tag in knee_tags:
            if f'{tag}_knee_work' in S and len(S):
                d[f'{tag}_knee_work_macro'] = float(S[f'{tag}_knee_work'].mean()); d[f'{tag}_knee_work_pooled'] = pooled(S[f'{tag}_knee_work'], S.N)
                d[f'{tag}_knee_reliable'] = prop(int(S[f'{tag}_knee_reliable'].sum()), len(S))
        out[lab] = d
    return out

def paired_rules(T, knee_tag):
    rel_d = T.thr_reliable - T[f'{knee_tag}_knee_reliable']; work_d = T.thr_work - T[f'{knee_tag}_knee_work']
    return {'reliability_diff_threshold_minus_knee': float(rel_d.mean()), 'reliability_diff_boot95': boot_mean(rel_d), 'reviews_threshold_reliable_knee_not': int(((T.thr_reliable == 1) & (T[f'{knee_tag}_knee_reliable'] == 0)).sum()), 'reviews_knee_reliable_threshold_not': int(((T.thr_reliable == 0) & (T[f'{knee_tag}_knee_reliable'] == 1)).sum()),
            'work_diff_threshold_minus_knee_macro': float(work_d.mean()), 'work_diff_boot95': boot_mean(work_d), 'reviews_threshold_reads_less': int((work_d < 0).sum()), 'reviews_threshold_reads_more': int((work_d > 0).sum())}

def bands(T):
    out = {}
    for b in ['0.03_0.07', '0.01_0.07']:
        out[b] = {'records_macro_frac': float(T[f'band_{b}_records_frac'].mean()), 'records_pooled_frac': float(T[f'band_{b}_records_n'].sum() / T.N.sum()), 'records_n': int(T[f'band_{b}_records_n'].sum()),
                  'included_macro_frac': float(T[f'band_{b}_included_frac'].mean()), 'included_pooled_frac': float(T[f'band_{b}_included_n'].sum() / T.n1.sum()), 'included_n': int(T[f'band_{b}_included_n'].sum()),
                  'reviews_with_included_in_band': int((T[f'band_{b}_included_n'] > 0).sum())}
    out['included_below_tau_n'] = int(T.n_included_below_tau.sum()); out['included_total'] = int(T.n1.sum()); out['records_total'] = int(T.N.sum())
    return out

out = {'analysis_id': 'AN-0001-04', 'started': started, 'tau': TAU, 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'executable': sys.executable},
       'settings': {'bootstrap': {'resamples': B, 'seed': BOOT_SEED, 'type': 'percentile CI of the mean over resampled reviews'}, 'wilson_z': Z, 'knee_reliability_primary': 'indicator on the seed-mean knee recall per review (as in synergy/final_eval.py and clef_eval.py)', 'knee_reliability_secondary': 'mean over seeds of the per-seed indicator (knee applied per seed)', 'estimands': 'macro = unweighted mean over reviews; pooled = record-weighted (sum of records read / sum of records; sum of included found / sum of included)'},
       'collections': {}}
J02 = pd.read_csv(AN02)
# ---------------- held-out SYNERGY (23)
by = load_by(SYN + '/test_records.json'); J = load_jev(SYN + '/jev_test.json'); T = threshold_rows(by, J)
T = T.join(al_columns(SYN + '/asr_test.json', [('asr_prior', 'asr'), ('asr_jevblend_3', 'hybrid'), ('asr_pseudo_10_50', 'pseudo'), ('h3_prior', 'h3')]))
T = T.join(al_columns(SYN + '/al_test_prereg.json', [('al|lr|oracle|0|0', 'lr'), ('al|nb|oracle|0|0', 'nb')]))
Z2 = pd.read_csv(SYN + '/zs_auc_test.csv').set_index('review')[['tfidf', 'bm25', 'minilm', 'bge']].add_prefix('auc_'); T = T.join(Z2)
jo = J02[J02.collection == 'heldout_final'].set_index('review')[['jevonly_wss95', 'jevonly_wss100', 'jevonly_stable_wss95', 'jevonly_stable_wss100']]; T = T.join(jo)
ta = pd.read_csv(SYN + '/p3_ta_auc.csv'); ta = ta[ta.split == 'test'].set_index('review')[['ta_pos', 'jev_ta', 'bge_ta']].rename(columns={'ta_pos': 'n1_ta', 'jev_ta': 'jev_auc_ta', 'bge_ta': 'bge_auc_ta'}); T = T.join(ta)
H = {'n_reviews': int(len(T)), 'records': int(T.N.sum()), 'included': int(T.n1.sum())}
H['threshold'] = rule_summary(T, 'thr_work', 'thr_rec', 'thr_reliable', 'Jev probability >= 0.07 (label-free)')
for tag, nm in [('asr', 'knee on ASReview default ranking (10 seeds)'), ('hybrid', 'knee on hybrid ranking (1 run)'), ('pseudo', 'knee on weak-supervision ranking (1 run)'), ('h3', 'knee on ASReview strongest preset (10 seeds)'), ('lr', 'knee on prespecified LR active learning (10 seeds)'), ('nb', 'knee on prespecified NB active learning (10 seeds)')]:
    if f'{tag}_knee_work' in T: H[f'knee_{tag}'] = rule_summary(T, f'{tag}_knee_work', f'{tag}_knee_rec', f'{tag}_knee_reliable', nm, f'{tag}_knee_rel_perseed')
H['paired_threshold_vs_knee_asr'] = paired_rules(T, 'asr'); H['paired_threshold_vs_knee_hybrid'] = paired_rules(T, 'hybrid')
H['size_restriction_150'] = size_restriction(T, ['asr', 'hybrid', 'lr', 'nb'], 150); H['size_restriction_500'] = size_restriction(T, ['asr', 'hybrid', 'lr', 'nb'], 500)
H['strata'], H['prevalence_tertile_cuts'] = strata(T, 'asr'); H['bands_below_tau'] = bands(T)
H['exact_ci_for_23_of_23'] = {'wilson95': wilson(23, 23), 'clopper_pearson95': clopper(23, 23), 'clopper_pearson_lower_formula': '0.05**(1/23)', 'value': 0.05 ** (1 / 23)}
out['collections']['heldout'] = H; T.reset_index().to_csv(HERE + '/per_review_heldout.csv', index=False)
# ---------------- CLEF 2019 (28 with final labels; 31 with title-abstract labels)
byc = load_by(CLEF + '/clef_records.json'); Jc = load_jev(SYN + '/jev_clef.json'); C = threshold_rows(byc, Jc)
C = C.join(al_columns(SYN + '/asr_clef.json', [('asr_prior', 'asr'), ('asr_jevblend_3', 'hybrid'), ('asr_pseudo_10_50', 'pseudo')]))
Zc = pd.read_csv(SYN + '/zs_auc_clef.csv').set_index('review')[['tfidf', 'bm25', 'minilm', 'bge']].add_prefix('auc_'); C = C.join(Zc)
joc = J02[J02.collection == 'clef_final'].set_index('review')[['jevonly_wss95', 'jevonly_wss100', 'jevonly_stable_wss95', 'jevonly_stable_wss100']]; C = C.join(joc)
typ = {k: v[0]['split'] for k, v in byc.items()}; C['review_type'] = [typ[k] for k in C.index]
K = {'n_reviews': int(len(C)), 'records': int(C.N.sum()), 'included': int(C.n1.sum())}
K['threshold'] = rule_summary(C, 'thr_work', 'thr_rec', 'thr_reliable', 'Jev probability >= 0.07 (label-free)')
for tag, nm in [('asr', 'knee on ASReview default ranking (10 seeds)'), ('hybrid', 'knee on hybrid ranking (1 run)'), ('pseudo', 'knee on weak-supervision ranking (1 run)')]:
    K[f'knee_{tag}'] = rule_summary(C, f'{tag}_knee_work', f'{tag}_knee_rec', f'{tag}_knee_reliable', nm, f'{tag}_knee_rel_perseed')
K['paired_threshold_vs_knee_asr'] = paired_rules(C, 'asr'); K['paired_threshold_vs_knee_hybrid'] = paired_rules(C, 'hybrid')
K['size_restriction_150'] = size_restriction(C, ['asr', 'hybrid', 'pseudo'], 150); K['size_restriction_500'] = size_restriction(C, ['asr', 'hybrid', 'pseudo'], 500)
K['strata'], K['prevalence_tertile_cuts'] = strata(C, 'asr'); K['bands_below_tau'] = bands(C)
K['by_review_type'] = {t: {'n': int(len(S)), 'thr_reliable': prop(int(S.thr_reliable.sum()), len(S)), 'thr_work_macro': float(S.thr_work.mean()), 'knee_asr_reliable': prop(int(S.asr_knee_reliable.sum()), len(S)), 'knee_asr_work_macro': float(S.asr_knee_work.mean())} for t, S in C.groupby('review_type')}
out['collections']['clef'] = K
# CLEF per-review table with the 3 title-abstract-only reviews appended
Tt = pd.read_csv(SYN + '/clef_zs_label_ta.csv').set_index('topic'); extra = []
for k in Tt.index:
    if k not in C.index:
        v = byc[k]; extra.append({'review': k, 'review_type': v[0]['split'], 'N': len(v), 'n1': 0, 'note': 'no included study among retrievable records; title-abstract labels only'})
Cout = C.reset_index(); Cout['n1_ta'] = [int(Tt.loc[k, 'n1']) if k in Tt.index else None for k in Cout.review]; Cout['jev_auc_ta'] = [float(Tt.loc[k, 'jev_AUC']) if k in Tt.index else None for k in Cout.review]; Cout['bge_auc_ta'] = [float(Tt.loc[k, 'bge_AUC']) if k in Tt.index else None for k in Cout.review]
E = pd.DataFrame(extra); E['n1_ta'] = [int(Tt.loc[k, 'n1']) for k in E.review]; E['jev_auc_ta'] = [float(Tt.loc[k, 'jev_AUC']) for k in E.review]; E['bge_auc_ta'] = [float(Tt.loc[k, 'bge_AUC']) for k in E.review]
pd.concat([Cout, E], ignore_index=True).to_csv(HERE + '/per_review_clef.csv', index=False)
# ---------------- development SYNERGY (73): threshold and knee on ASReview only
byd = load_by(SYN + '/dev2000_records.json'); Jd = load_jev(SYN + '/jev_dev2000.json'); Dv = threshold_rows(byd, Jd)
Dv = Dv.join(al_columns(SYN + '/asr_dev.json', [('asr_prior', 'asr'), ('asr_jevblend_3', 'hybrid')]))
Dd = {'n_reviews': int(len(Dv)), 'records': int(Dv.N.sum()), 'included': int(Dv.n1.sum())}
Dd['threshold'] = rule_summary(Dv, 'thr_work', 'thr_rec', 'thr_reliable', 'Jev probability >= 0.07 (label-free; tau was selected on these reviews)')
for tag, nm in [('asr', 'knee on ASReview default ranking (10 seeds)'), ('hybrid', 'knee on hybrid ranking (1 run)')]:
    if f'{tag}_knee_work' in Dv: Dd[f'knee_{tag}'] = rule_summary(Dv, f'{tag}_knee_work', f'{tag}_knee_rec', f'{tag}_knee_reliable', nm, f'{tag}_knee_rel_perseed')
Dd['paired_threshold_vs_knee_asr'] = paired_rules(Dv, 'asr'); Dd['size_restriction_150'] = size_restriction(Dv, ['asr'], 150); Dd['strata'], Dd['prevalence_tertile_cuts'] = strata(Dv, 'asr'); Dd['bands_below_tau'] = bands(Dv)
out['collections']['development'] = Dd; Dv.reset_index().to_csv(HERE + '/per_review_development.csv', index=False)
# ---------------- combined band summary and v0 comparison
allT = pd.concat([T.assign(collection='heldout'), C.assign(collection='clef')]); out['bands_heldout_plus_clef'] = bands(allT)
v0 = {'heldout_thr_reliable': '23 of 23', 'heldout_thr_work': 0.589, 'heldout_thr_mean_recall': 0.998, 'heldout_thr_min_recall': 0.958, 'heldout_knee_asr_reliable': '20 of 23', 'heldout_knee_asr_work': 0.940, 'heldout_knee_hybrid_reliable': '23 of 23', 'heldout_knee_hybrid_work': 0.837,
      'heldout_knee_lr_nb_reliable': '21 of 23 each', 'heldout_knee_lr_work': 0.961, 'heldout_knee_nb_work': 0.963, 'clef_thr_reliable': '25 of 28', 'clef_thr_mean_recall': 0.990, 'clef_thr_min_recall': 0.889, 'clef_thr_work': 0.391, 'clef_knee_asr_reliable': '26 of 28', 'clef_knee_asr_work': 0.760,
      'clef_knee_hybrid_reliable': '27 of 28', 'clef_knee_hybrid_work': 0.598, 'clef_knee_pseudo_reliable': '27 of 28', 'clef_knee_pseudo_work': 0.819, 'dev_thr_reliability': 0.986, 'dev_thr_work': 0.583}
new = {'heldout_thr_reliable': f"{H['threshold']['reliability']['k']} of {H['threshold']['reliability']['n']}", 'heldout_thr_work': round(H['threshold']['work_macro']['mean'], 3), 'heldout_thr_mean_recall': round(H['threshold']['recall_macro']['mean'], 3), 'heldout_thr_min_recall': round(H['threshold']['recall_macro']['min'], 3),
       'heldout_knee_asr_reliable': f"{H['knee_asr']['reliability']['k']} of {H['knee_asr']['reliability']['n']}", 'heldout_knee_asr_work': round(H['knee_asr']['work_macro']['mean'], 3), 'heldout_knee_hybrid_reliable': f"{H['knee_hybrid']['reliability']['k']} of {H['knee_hybrid']['reliability']['n']}", 'heldout_knee_hybrid_work': round(H['knee_hybrid']['work_macro']['mean'], 3),
       'heldout_knee_lr_nb_reliable': f"{H['knee_lr']['reliability']['k']} and {H['knee_nb']['reliability']['k']} of 23", 'heldout_knee_lr_work': round(H['knee_lr']['work_macro']['mean'], 3), 'heldout_knee_nb_work': round(H['knee_nb']['work_macro']['mean'], 3),
       'clef_thr_reliable': f"{K['threshold']['reliability']['k']} of {K['threshold']['reliability']['n']}", 'clef_thr_mean_recall': round(K['threshold']['recall_macro']['mean'], 3), 'clef_thr_min_recall': round(K['threshold']['recall_macro']['min'], 3), 'clef_thr_work': round(K['threshold']['work_macro']['mean'], 3),
       'clef_knee_asr_reliable': f"{K['knee_asr']['reliability']['k']} of {K['knee_asr']['reliability']['n']}", 'clef_knee_asr_work': round(K['knee_asr']['work_macro']['mean'], 3), 'clef_knee_hybrid_reliable': f"{K['knee_hybrid']['reliability']['k']} of {K['knee_hybrid']['reliability']['n']}", 'clef_knee_hybrid_work': round(K['knee_hybrid']['work_macro']['mean'], 3),
       'clef_knee_pseudo_reliable': f"{K['knee_pseudo']['reliability']['k']} of {K['knee_pseudo']['reliability']['n']}", 'clef_knee_pseudo_work': round(K['knee_pseudo']['work_macro']['mean'], 3), 'dev_thr_reliability': round(Dd['threshold']['reliability']['proportion'], 3), 'dev_thr_work': round(Dd['threshold']['work_macro']['mean'], 3)}
out['v0_comparison'] = {k: {'v0': v0[k], 'new': new[k], 'match': v0[k] == new[k]} for k in v0}
out['elapsed_seconds'] = round(time.time() - t0, 1); out['finished'] = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
json.dump(out, open(HERE + '/results.json', 'w'), indent=1, default=float)
srows = []
for cn in ['heldout', 'clef', 'development']:
    for r in out['collections'][cn]['strata']: srows.append({'collection': cn, **r})
pd.DataFrame(srows).to_csv(HERE + '/strata.csv', index=False)
log('v0 comparison:'); log(json.dumps(out['v0_comparison'], indent=1))
for cn in ['heldout', 'clef', 'development']:
    c = out['collections'][cn]; log(f'== {cn}: reviews {c["n_reviews"]}, records {c["records"]}, included {c["included"]}')
    for key in [k for k in c if k == 'threshold' or k.startswith('knee_')]:
        s = c[key]; r = s['reliability']
        log(f"  {key}: {r['k']}/{r['n']} = {r['proportion']:.4f} Wilson {[round(x, 4) for x in r['wilson95']]} CP {[round(x, 4) for x in r['clopper_pearson95']]}; work macro {s['work_macro']['mean']:.4f} boot {[round(x, 4) for x in s['work_macro']['boot95_of_mean']]} median {s['work_macro']['median']:.4f} IQR {s['work_macro']['q1']:.4f}-{s['work_macro']['q3']:.4f} range {s['work_macro']['min']:.4f}-{s['work_macro']['max']:.4f}; pooled {s['work_pooled_record_weighted']:.4f}; recall macro {s['recall_macro']['mean']:.4f} min {s['recall_macro']['min']:.4f}" + (f"; pooled recall {s['recall_pooled']:.4f}" if 'recall_pooled' in s else '') + (f"; per-seed reliability {s['reliability_per_seed_mean']:.4f}" if 'reliability_per_seed_mean' in s else ''))
        if s['failing_reviews']: log('     failing:', [(f['review'], round(f['recall'], 3)) for f in s['failing_reviews']])
    log('  paired thr vs knee asr:', json.dumps(c['paired_threshold_vs_knee_asr']))
    log('  size restriction 150:', json.dumps(c['size_restriction_150'], default=float)[:1500])
    log('  bands:', json.dumps(c['bands_below_tau']))
    log('  tertile cuts:', c['prevalence_tertile_cuts'])
log('bands held-out plus CLEF:', json.dumps(out['bands_heldout_plus_clef']))
log(pd.DataFrame(srows).to_string())
log('elapsed', out['elapsed_seconds'], 's')
