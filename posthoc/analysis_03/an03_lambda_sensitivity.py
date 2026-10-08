#!/usr/bin/env python
# AN-0001-03: sensitivity of the hybrid (ASReview + Jev blend) to lambda = 5 and 10, and the limiting case of the
# pure Jev ranking. Post hoc; the primary hybrid stays lambda = 3 (INTEGRITY_RULES I-4).
# Inputs (read only): new simulation outputs in this folder (an03_asr_{dev,heldout,clef}.json; methods asr_jevblend_5
# and asr_jevblend_10, one run each), the stored original simulations (synergy/asr_dev.json, asr_test.json, asr_clef.json),
# the record and Jev score files (for the Jev-only ranking, computed as in AN-0001-02), and AN-0001-02/per_review.csv
# (consistency check of the Jev-only values on the held-out and CLEF reviews).
import os, sys, json, time, subprocess
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from al_sim import metrics   # unchanged copy of synergy/al_sim.py

ROOT = '/N/project/AiLab/jev'
HERE = os.path.dirname(os.path.abspath(__file__))
AN02 = os.path.join(ROOT, 'review_pipeline/rounds/round_0001/revision/analysis/AN-0001-02/per_review.csv')
MET = ['wss95', 'wss100', 'rec@10', 'rec@20', 'rec@30', 'knee_work', 'knee_rec']
BOOT_SEED = 1003; B = 5000; TIE_SEEDS = list(range(10))
def now(): return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S %Z'], capture_output=True, text=True).stdout.strip()

COLL = {
    'development': dict(records='synergy/dev2000_records.json', jev='synergy/jev_dev2000.json', stored='synergy/asr_dev.json', new='an03_asr_dev.json', an02=None),
    'heldout': dict(records='synergy/test_records.json', jev='synergy/jev_test.json', stored='synergy/asr_test.json', new='an03_asr_heldout.json', an02='heldout_final'),
    'clef': dict(records='clef/clef_records.json', jev='synergy/jev_clef.json', stored='synergy/asr_clef.json', new='an03_asr_clef.json', an02='clef_final'),
}

def jev_only(records, jev):
    """Jev-only ranking per review: descending probability, ties broken by a uniform random key (as AN-0001-02); mean over seeds 0-9."""
    R = json.load(open(os.path.join(ROOT, records)))
    J = {d['id']: d['p'] for d in json.load(open(os.path.join(ROOT, jev))) if d.get('ok') and d.get('p') is not None}
    by = {}
    for r in R: by.setdefault(r['review'], []).append(r)
    rows = {}
    for k, rs in by.items():
        if not all(r['id'] in J for r in rs): continue
        y = np.array([r['label'] for r in rs]); p = np.array([J[r['id']] for r in rs], float)
        if not (0 < y.sum() < len(y)): continue
        acc = {c: [] for c in ['wss95', 'wss100', 'rec@10', 'rec@20', 'rec@30']}
        for s in TIE_SEEDS:
            rng = np.random.default_rng(s); seq = np.lexsort((rng.random(len(p)), -p)); m = metrics(seq, y)
            for c in acc: acc[c].append(m[c])
        rows[k] = {c: float(np.mean(v)) for c, v in acc.items()}
    return rows

def paired(x, y, label):
    d = np.asarray(x, float) - np.asarray(y, float); n = len(d)
    rng = np.random.default_rng(BOOT_SEED); idx = rng.integers(0, n, (B, n)); m = d[idx].mean(1)
    out = {'n': n, 'mean_diff': float(d.mean()), 'ci95': [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
           'wins': int((d > 0).sum()), 'losses': int((d < 0).sum()), 'ties': int((d == 0).sum()), 'label': label}
    if n > 5 and np.any(d != 0):
        out['wilcoxon_p'] = float(wilcoxon(x, y).pvalue)
        try: out['wilcoxon_p_exact'] = float(wilcoxon(x, y, method='exact').pvalue)
        except Exception as e: out['wilcoxon_p_exact'] = None
        out['wilcoxon_p_asymptotic'] = float(wilcoxon(x, y, method='asymptotic').pvalue)
        nz = int((d != 0).sum()); ties = len(np.unique(np.abs(d[d != 0]))) < nz
        out['wilcoxon_method_auto'] = 'exact' if (n <= 50 and out['ties'] == 0 and not ties) else ('exact_enumeration' if n <= 13 else 'asymptotic')
    else:
        out['wilcoxon_p'] = None
    return out

def main():
    t0 = time.time(); started = now()
    print('AN-0001-03 comparison started', started, flush=True)
    res = {'analysis_id': 'AN-0001-03', 'started': started, 'settings': {
        'primary_hybrid': 'asr_jevblend_3 (unchanged, I-4)', 'new_methods': ['asr_jevblend_5', 'asr_jevblend_10'], 'runs_per_method': 1,
        'jev_only': 'descending Jev probability, ties by uniform random key np.lexsort((default_rng(seed).random(N), -p)), seeds 0-9, mean (as AN-0001-02)',
        'bootstrap': {'resamples': B, 'seed': BOOT_SEED, 'type': 'percentile, reviews resampled with replacement, mean of paired differences'},
        'wilcoxon': 'scipy.stats.wilcoxon defaults (two-sided, zero_method wilcox, method auto); exact and asymptotic p also stored',
        'metrics': MET}, 'collections': {}}
    per_rows = []
    for name, c in COLL.items():
        new = pd.DataFrame(json.load(open(os.path.join(HERE, c['new']))))
        stored = pd.DataFrame(json.load(open(os.path.join(ROOT, c['stored']))))
        new = new[new.wss95.notna()]; stored = stored[stored.wss95.notna()]
        A = pd.concat([stored, new], ignore_index=True)
        g = A.groupby(['review', 'method'])[MET].mean().unstack('method')
        lams = [m for m in ['asr_jevblend_1', 'asr_jevblend_2', 'asr_jevblend_3', 'asr_jevblend_5', 'asr_jevblend_10'] if ('wss95', m) in g]
        jo = jev_only(c['records'], c['jev'])
        revs = [r for r in g.index if r in jo]
        assert len(revs) == len(g.index), (name, len(revs), len(g.index))
        g = g.loc[revs]
        JO = pd.DataFrame(jo).T.loc[revs]
        coll = {'n_reviews': len(revs), 'reviews': revs, 'methods_present': lams,
                'tasks_new': {m: int((new.method == m).sum()) for m in ['asr_jevblend_5', 'asr_jevblend_10']},
                'seconds_new_runs_sum': float(new.secs.sum()),
                'macro': {m: {c_: float(g[(c_, m)].mean()) for c_ in MET} for m in lams + ['asr_prior']},
                'macro_jev_only': {c_: float(JO[c_].mean()) for c_ in JO.columns},
                'comparisons': {}}
        # consistency check against AN-0001-02 (held-out, CLEF)
        if c['an02']:
            P = pd.read_csv(AN02); P = P[P.collection == c['an02']].set_index('review')
            common = [r for r in revs if r in P.index]
            diffs = {c_: float(np.max(np.abs(JO.loc[common, c_].values - P.loc[common, f'jevonly_{c_}'].values))) for c_ in JO.columns}
            coll['jev_only_check_vs_AN02_max_abs_diff'] = diffs; coll['jev_only_check_n'] = len(common)
        for m in lams:
            if m == 'asr_jevblend_3': continue
            coll['comparisons'][f'{m}_minus_asr_jevblend_3'] = {c_: paired(g[(c_, m)].values, g[(c_, 'asr_jevblend_3')].values, f'{m} minus lambda=3: {c_}') for c_ in MET}
        for m in lams:
            coll['comparisons'][f'{m}_minus_jev_only'] = {c_: paired(g[(c_, m)].values, JO[c_].values, f'{m} minus Jev-only: {c_}') for c_ in JO.columns}
            coll['comparisons'][f'{m}_minus_asr_prior'] = {c_: paired(g[(c_, m)].values, g[(c_, 'asr_prior')].values, f'{m} minus ASReview: {c_}') for c_ in MET}
        coll['comparisons']['jev_only_minus_asr_prior'] = {c_: paired(JO[c_].values, g[(c_, 'asr_prior')].values, f'Jev-only minus ASReview: {c_}') for c_ in JO.columns}
        # selection rule check: largest macro WSS@95 over the grid
        w = {m: coll['macro'][m]['wss95'] for m in lams}
        coll['selection_rule_max_wss95'] = {'grid': lams, 'macro_wss95': w, 'argmax': max(w, key=w.get),
                                            'original_grid_argmax': max({m: w[m] for m in lams if m in ('asr_jevblend_1', 'asr_jevblend_2', 'asr_jevblend_3')}, key=w.get)}
        # per-review wins of larger lambda over lambda=3 by review size
        for r in revs:
            row = {'collection': name, 'review': r, 'N': int(g.loc[r, ('N', lams[0])]) if ('N', lams[0]) in g else None}
            for m in lams:
                for c_ in MET: row[f'{m}_{c_}'] = float(g.loc[r, (c_, m)])
            for c_ in MET: row[f'asr_prior_{c_}'] = float(g.loc[r, (c_, 'asr_prior')])
            for c_ in JO.columns: row[f'jev_only_{c_}'] = float(JO.loc[r, c_])
            per_rows.append(row)
        res['collections'][name] = coll
        print(name, 'reviews', len(revs), 'macro wss95', {m: round(w[m], 4) for m in w}, 'jev_only', round(coll['macro_jev_only']['wss95'], 4), flush=True)
    # N per review from the records is not in g when methods differ; fill from stored file
    for name, c in COLL.items():
        stored = pd.DataFrame(json.load(open(os.path.join(ROOT, c['stored']))))
        nmap = stored.groupby('review').N.first().to_dict()
        for row in per_rows:
            if row['collection'] == name: row['N'] = int(nmap[row['review']])
    pd.DataFrame(per_rows).to_csv(os.path.join(HERE, 'per_review_lambda.csv'), index=False)
    res['elapsed_seconds'] = time.time() - t0; res['finished'] = now()
    json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
    print('finished', res['finished'], f'{res["elapsed_seconds"]:.1f}s', flush=True)

if __name__ == '__main__':
    main()
