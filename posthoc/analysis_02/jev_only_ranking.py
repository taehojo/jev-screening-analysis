#!/usr/bin/env python
# AN-0001-02: seed-fixed output for the label-free Jev ranking, and paired comparisons
# hybrid minus Jev-only and Jev-only minus ASReview on held-out, CLEF, pilot review and title-abstract labels.
# Reads original outputs only (read-only); writes into this folder only.
import os, sys, json, math, time, platform, datetime
import numpy as np, pandas as pd, scipy, sklearn
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from al_sim_copy import metrics   # unchanged copy of synergy/al_sim.py (md5 in README)
ROOT = '/N/project/AiLab/jev'; SYN = ROOT + '/synergy'
SEEDS = list(range(10)); B = 5000; BOOT_SEED = 1002
MET = ['wss95', 'wss100', 'rec@10', 'rec@20', 'rec@30']
EXTRA = ['map', 'tnr95']
t0 = time.time(); started = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
def log(*a): print(*a, flush=True)

def rank_extra(seq, y):
    # MAP and TNR@95 with the definitions of synergy/clef_eval.py rank_metrics(), applied to a given reading order
    y = np.asarray(y); ys = y[np.asarray(seq)]; n1 = int(ys.sum()); N = len(y)
    prec = np.cumsum(ys) / np.arange(1, N + 1); ap = float((prec * ys).sum() / n1)
    k95 = int(np.searchsorted(np.cumsum(ys), math.ceil(0.95 * n1)) + 1)
    tnr95 = ((N - k95) - (n1 - ys[:k95].sum())) / (N - n1)
    return {'map': ap, 'tnr95': float(tnr95)}

def order_random(p, seed):
    rng = np.random.default_rng(seed); N = len(p)
    return np.lexsort((rng.random(N), -p))   # primary key descending p; ties broken by a uniform random key

def order_stable(p):
    return np.argsort(-p, kind='stable')     # ties kept in data-file order (as in clef_eval.py rank_metrics)

def load_by(path):
    by = {}
    for r in json.load(open(path)): by.setdefault(r['review'], []).append(r)
    return by

def load_jev(path):
    return {d['id']: d['p'] for d in json.load(open(path)) if d.get('ok') and d.get('p') is not None}

def al_table(path):
    A = pd.DataFrame([d for d in json.load(open(path)) if d.get('wss95') is not None])
    return A.groupby(['review', 'method'])[MET].mean().unstack('method')   # seed mean per review, then unstack

def boot_ci(d, seed=BOOT_SEED, Bn=B):
    d = np.asarray(d, float); rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (Bn, len(d)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

def wilcox(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); d = a - b; n = len(d)
    zeros = int((d == 0).sum()); dnz = np.abs(d[d != 0]); ties = int(len(dnz) - len(np.unique(dnz)))
    out = {'n': n, 'zeros': zeros, 'ties_among_nonzero_abs_diffs': ties}
    if n < 2 or (d != 0).sum() == 0:
        out.update({'p_auto': None, 'method_auto': 'not applicable'}); return out
    r = wilcoxon(a, b)
    out['statistic'] = float(r.statistic); out['p_auto'] = float(r.pvalue)
    out['method_auto'] = 'exact' if (n <= 50 and zeros == 0 and ties == 0) else ('exact_permutation_ties_or_zeros' if n <= 13 else 'asymptotic')   # SciPy 1.18.1 auto rule from the docstring Notes
    try: out['p_exact'] = float(wilcoxon(a, b, method='exact').pvalue)
    except Exception as e: out['p_exact'] = None; out['p_exact_error'] = repr(e)
    out['p_asymptotic'] = float(wilcoxon(a, b, method='asymptotic').pvalue)
    out['p_auto_equals_asymptotic'] = abs(out['p_auto'] - out['p_asymptotic']) < 1e-15
    out['p_auto_equals_exact'] = (out.get('p_exact') is not None and abs(out['p_auto'] - out['p_exact']) < 1e-15)
    return out

def fmt_p(p):
    if p is None: return None
    return '<0.0001' if p < 0.0001 else f'{p:.2g}'

def paired(a, b, name):
    a = pd.Series(a, dtype=float); b = pd.Series(b, dtype=float); ok = a.notna() & b.notna(); d = (a - b)[ok]
    res = {'compare': name, 'n': int(ok.sum()), 'mean_a': float(a[ok].mean()), 'mean_b': float(b[ok].mean()), 'diff': float(d.mean()),
           'ci95_percentile_bootstrap': boot_ci(d) if ok.sum() > 1 else None,
           'wins_a_gt_b': int((d > 0).sum()), 'losses_a_lt_b': int((d < 0).sum()), 'ties': int((d == 0).sum())}
    if ok.sum() > 5:
        w = wilcox(a[ok], b[ok]); res['wilcoxon'] = w; res['p_formatted'] = fmt_p(w.get('p_auto'))
    return res

def run_collection(name, by, J, al_path, label_key='label', reviews=None):
    log(f'== {name}')
    g = al_table(al_path) if al_path else None
    rows = []; per_seed = []
    revs = [k for k in by if all(r['id'] in J for r in by[k])]
    if reviews is not None: revs = [k for k in revs if k in set(reviews)]
    for k in revs:
        rs = by[k]
        if any(r.get(label_key) is None for r in rs): continue
        y = np.array([int(r[label_key]) for r in rs]); N = len(y); n1 = int(y.sum())
        if not (0 < n1 < N): continue
        p = np.array([float(J[r['id']]) for r in rs])
        row = {'collection': name, 'review': k, 'N': N, 'n1': n1, 'prevalence': n1 / N, 'n_distinct_p': int(len(np.unique(p)))}
        ms = []
        for s in SEEDS:
            seq = order_random(p, s); m = metrics(seq, y); m.update(rank_extra(seq, y)); ms.append(m)
            per_seed.append({'collection': name, 'review': k, 'seed': s, 'N': N, 'n1': n1, **{c: m[c] for c in ['k95', 'k100'] + MET + EXTRA}})
        for c in MET + EXTRA:
            vals = np.array([m[c] for m in ms])
            row[f'jevonly_{c}'] = float(vals.mean()); row[f'jevonly_{c}_seedmin'] = float(vals.min()); row[f'jevonly_{c}_seedmax'] = float(vals.max())
        seq = order_stable(p); m = metrics(seq, y); m.update(rank_extra(seq, y))
        for c in MET + EXTRA: row[f'jevonly_stable_{c}'] = float(m[c])
        if g is not None and k in g.index:
            for meth, tag in [('asr_prior', 'asr'), ('asr_jevblend_3', 'hybrid'), ('asr_jev', 'asr_jevstart'), ('asr_pseudo_10_50', 'pseudo'), ('asr_random', 'asr_random')]:
                if ('wss95', meth) in g.columns:
                    for c in MET:
                        v = g.loc[k, (c, meth)]; row[f'{tag}_{c}'] = float(v) if pd.notna(v) else None
        rows.append(row)
    T = pd.DataFrame(rows); P = pd.DataFrame(per_seed)
    macro_by_seed = P.groupby('seed')[MET + EXTRA].mean()
    res = {'n_reviews': int(len(T)), 'label': label_key, 'reviews': T.review.tolist(), 'records_total': int(T.N.sum()), 'included_total': int(T.n1.sum()),
           'jevonly_macro_random_ties_mean_of_10_seeds': {c: float(T[f'jevonly_{c}'].mean()) for c in MET + EXTRA},
           'jevonly_macro_random_ties_min_over_seeds': {c: float(macro_by_seed[c].min()) for c in MET + EXTRA},
           'jevonly_macro_random_ties_max_over_seeds': {c: float(macro_by_seed[c].max()) for c in MET + EXTRA},
           'jevonly_macro_stable_order': {c: float(T[f'jevonly_stable_{c}'].mean()) for c in MET + EXTRA}}
    for tag in ['asr', 'hybrid', 'asr_jevstart', 'pseudo', 'asr_random']:
        if f'{tag}_wss95' in T: res[f'{tag}_macro'] = {c: float(T[f'{tag}_{c}'].mean()) for c in MET}
    if len(T) > 1 and 'hybrid_wss95' in T:
        res['comparisons'] = {}
        pairs = [('hybrid', 'jevonly', 'hybrid minus Jev-only (random ties, 10-seed mean)'), ('jevonly', 'asr', 'Jev-only minus ASReview default (10-seed mean)'),
                 ('hybrid', 'asr', 'hybrid minus ASReview default (reproduction check of the stored comparison)')]
        if 'asr_jevstart_wss95' in T: pairs.append(('asr_jevstart', 'jevonly', 'ASReview started in Jev order without blending minus Jev-only'))
        if 'pseudo_wss95' in T: pairs.append(('pseudo', 'jevonly', 'LLM-guided weak supervision minus Jev-only'))
        for a, b, nm in pairs:
            res['comparisons'][f'{a}_minus_{b}'] = {c: paired(T[f'{a}_{c}'], T[f'{b}_{c}'], f'{nm}: {c}') for c in MET}
        res['comparisons']['hybrid_minus_jevonly_stable'] = {c: paired(T[f'hybrid_{c}'], T[f'jevonly_stable_{c}'], f'hybrid minus Jev-only (stable order): {c}') for c in MET}
    return res, T, P

if __name__ == '__main__':
    out = {'analysis_id': 'AN-0001-02', 'started': started, 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'executable': sys.executable},
           'settings': {'tie_break_seeds': SEEDS, 'tie_break': 'np.lexsort((default_rng(seed).random(N), -p)): descending probability, ties by uniform random key', 'stable_order': "np.argsort(-p, kind='stable')",
                        'bootstrap': {'resamples': B, 'seed': BOOT_SEED, 'type': 'percentile, resampling reviews with replacement, mean of paired differences'},
                        'wilcoxon': 'scipy.stats.wilcoxon defaults (two-sided, zero_method wilcox, correction False, method auto); method_auto inferred from n<=50, zeros and ties',
                        'hybrid': 'asr_jevblend_3 seed 0 (single run, as stored)', 'asreview': 'asr_prior, mean of seeds 0-9 per review', 'metrics': 'metrics() of synergy/al_sim.py; MAP and TNR@95 as in synergy/clef_eval.py rank_metrics()'},
           'collections': {}}
    tables = []; seeds_tables = []
    # 1. held-out SYNERGY reviews, final inclusion labels
    by = load_by(SYN + '/test_records.json'); J = load_jev(SYN + '/jev_test.json')
    r, T, P = run_collection('heldout_final', by, J, SYN + '/asr_test.json'); out['collections']['heldout_final'] = r; tables.append(T); seeds_tables.append(P)
    # 2. held-out, title and abstract labels (12 reviews with labels; AL results in asr_test_ta.json)
    ta_revs = sorted({d['review'] for d in json.load(open(SYN + '/asr_test_ta.json'))})
    r, T, P = run_collection('heldout_ta', by, J, SYN + '/asr_test_ta.json', label_key='label_ta', reviews=ta_revs); out['collections']['heldout_ta'] = r; tables.append(T); seeds_tables.append(P)
    # 3. CLEF 2019 Cochrane reviews, final inclusion (28 reviews with AL results)
    byc = load_by(ROOT + '/clef/clef_records.json'); Jc = load_jev(SYN + '/jev_clef.json')
    clef_revs = sorted({d['review'] for d in json.load(open(SYN + '/asr_clef.json'))})
    r, T, P = run_collection('clef_final', byc, Jc, SYN + '/asr_clef.json', reviews=clef_revs); out['collections']['clef_final'] = r; tables.append(T); seeds_tables.append(P)
    # 3b. CLEF title and abstract labels, Jev-only only (31 reviews; no AL run exists at this label level) - descriptive addition
    r, T, P = run_collection('clef_ta_jevonly_only', byc, Jc, None, label_key='label_ta'); out['collections']['clef_ta_jevonly_only'] = r; tables.append(T); seeds_tables.append(P)
    # 4. pilot review (Alzheimer's disease review, PubMed arm, title-abstract decisions as label); aggregate values only (I-14)
    byd = load_by(SYN + '/dhl_records.json'); Jd = load_jev(SYN + '/dhl_jev_b10.json')
    r, T, P = run_collection('pilot_review', byd, Jd, SYN + '/asr_dhl.json'); r.pop('reviews', None); out['collections']['pilot_review'] = r
    Tp = T.drop(columns=['review']).assign(review='pilot'); tables.append(Tp); seeds_tables.append(P.assign(review='pilot'))
    # comparison with v0 values (RESULTS.md and appendix section 6): held-out Jev-only WSS@95 0.788, WSS@100 0.801, MAP 0.513, TNR@95 0.869
    h = out['collections']['heldout_final']['jevonly_macro_random_ties_mean_of_10_seeds']
    v0 = {'wss95': 0.788, 'wss100': 0.801, 'map': 0.513, 'tnr95': 0.869}
    out['v0_comparison_heldout_jevonly'] = {c: {'v0': v0[c], 'new_random_ties_10_seed_mean': round(h[c], 4), 'new_rounded_3dp': round(h[c], 3), 'match_at_3dp': bool(round(h[c], 3) == v0[c]),
                                                'stable_order': round(out['collections']['heldout_final']['jevonly_macro_stable_order'][c], 4)} for c in v0}
    out['elapsed_seconds'] = round(time.time() - t0, 1); out['finished'] = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
    pd.concat(tables, ignore_index=True).to_csv(HERE + '/per_review.csv', index=False)
    pd.concat(seeds_tables, ignore_index=True).to_csv(HERE + '/jev_only_per_seed.csv', index=False)
    json.dump(out, open(HERE + '/results.json', 'w'), indent=1, default=float)
    log(json.dumps(out['v0_comparison_heldout_jevonly'], indent=1))
    for nm, r in out['collections'].items():
        log(nm, 'n=', r['n_reviews'], 'Jev-only macro', {c: round(v, 4) for c, v in r['jevonly_macro_random_ties_mean_of_10_seeds'].items()})
        for key in ['hybrid_macro', 'asr_macro']:
            if key in r: log('  ', key, {c: round(v, 4) for c, v in r[key].items()})
        if 'comparisons' in r:
            for cmp_, d in r['comparisons'].items():
                for c, v in d.items():
                    log(f"   {cmp_} {c}: diff={v['diff']:.4f} CI={[round(x, 4) for x in v['ci95_percentile_bootstrap']]} wins/losses/ties={v['wins_a_gt_b']}/{v['losses_a_lt_b']}/{v['ties']} p={v.get('p_formatted')} ({v.get('wilcoxon', {}).get('method_auto')})")
    log('elapsed', out['elapsed_seconds'], 's')
