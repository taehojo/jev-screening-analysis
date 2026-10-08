#!/usr/bin/env python
# npj AN-0001-04 (post hoc): true negative rate at 95% recall (TNR@95; normalised WSS@95 of Kusa et al. 2023) for the primary
# ranking comparisons, and the rounding of k95. Reads stored outputs only; writes only into this folder. No model call.
# Orders: ASReview default (asr_prior seeds 0-9, Lancet AN-0001-08 re-run orders), hybrid (asr_jevblend_3 seeds 0-9, npj AN-0001-03),
# Jev ranking (random ties seeds 0-9, as Lancet AN-0001-02; also data-file order), bge-base scores (synergy/scores_zs).
import os, json, math, time, subprocess, platform, warnings
from fractions import Fraction as F
import numpy as np, pandas as pd, scipy
from scipy.stats import wilcoxon

ROOT = '/N/project/AiLab/jev'; SYN = ROOT + '/synergy'
LAN = ROOT + '/review_pipeline/rounds/round_0001/revision/analysis'
NPJ = ROOT + '/review_pipeline_npj/rounds/round_0001/revision/analysis'
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = list(range(10)); B = 5000; BOOT_SEED = 20260928
def now(): return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S %Z'], capture_output=True, text=True).stdout.strip()
def J(p): return json.load(open(p))
def log(*a): print(*a, flush=True)
COLL = {'heldout': dict(records=SYN + '/test_records.json', jev=SYN + '/jev_test.json', stored=SYN + '/asr_test.json', zs='test',
                        asr_orders=LAN + '/AN-0001-08/an08_asr_heldout_orders.json', hyb_orders=NPJ + '/AN-0001-03/an03_hybrid_heldout_orders.json'),
        'clef': dict(records=ROOT + '/clef/clef_records.json', jev=SYN + '/jev_clef.json', stored=SYN + '/asr_clef.json', zs='clef',
                     asr_orders=LAN + '/AN-0001-08/an08_asr_clef_orders.json', hyb_orders=NPJ + '/AN-0001-03/an03_hybrid_clef_orders.json')}

def tnr95(seq, y):
    """TNR@95 as a fraction: excluded records not read when the ceil(0.95 n1)-th included record is found / all excluded records.
    k95 as in synergy/al_sim.py metrics() (pos[ceil(0.95*n1)-1]); identical to synergy/clef_eval.py rank_metrics() for a given order."""
    y = np.asarray(y); ys = y[np.asarray(seq)]; N = len(y); n1 = int(y.sum()); n0 = N - n1
    need = math.ceil(0.95 * n1); pos = np.where(ys == 1)[0] + 1; k95 = int(pos[need - 1])
    fp = k95 - need   # excluded records read up to and including position k95
    return F(n0 - fp, n0), k95, need

def mean_f(v): v = list(v); return sum(v, F(0)) / len(v)
def boot_ci(d):
    d = np.asarray(d, float); rng = np.random.default_rng(BOOT_SEED); m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
def method_rule(d):
    nz = d[d != 0]; ties = len(nz) - len(np.unique(np.abs(nz))); zeros = int((d == 0).sum())
    if zeros == 0 and ties == 0 and len(d) <= 50: return 'exact'
    if len(d) <= 13: return 'exact (permutation; zero or tied differences)'
    return 'asymptotic'
def paired(a, b, label):
    d = [x - y for x, y in zip(a, b)]; dfl = np.array([float(v) for v in d])
    out = {'label': label, 'n': len(d), 'mean_a': float(mean_f(a)), 'mean_b': float(mean_f(b)), 'mean_diff': float(mean_f(d)), 'ci95_bootstrap': boot_ci(dfl),
           'higher': int(sum(v > 0 for v in d)), 'lower': int(sum(v < 0 for v in d)), 'tied': int(sum(v == 0 for v in d)), 'wilcoxon_p': None, 'wilcoxon_method': None}
    if np.any(dfl != 0):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); out['wilcoxon_p'] = float(wilcoxon(dfl).pvalue)
        out['wilcoxon_method'] = method_rule(dfl)
    return out

def main():
    t0 = time.time()
    res = {'analysis_id': 'AN-0001-04 (npj round 1)', 'post_hoc': True, 'started': now(),
           'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__},
           'definition': {'tnr95': 'TN/(TN+FP) at the point where the ceil(0.95 x n_included)-th included record is read: (excluded records not yet read) / (all excluded records)',
                          'k95': 'rank of the ceil(0.95 x n_included)-th included record (synergy/al_sim.py metrics(): pos[math.ceil(0.95 * n1) - 1])',
                          'bootstrap': f'percentile, {B} resamples of reviews, seed {BOOT_SEED}', 'per_review_value': 'mean over seeds 0-9 (exact fractions)'},
           'collections': {}}
    rows = []
    for coll, c in COLL.items():
        log('==', coll)
        stored = J(c['stored']); revs = sorted({d['review'] for d in stored if d['method'] == 'asr_jevblend_3' and d.get('wss95') is not None})
        by = {}
        for r in J(c['records']): by.setdefault(r['review'], []).append(r)
        P = {d['id']: d['p'] for d in J(c['jev']) if d.get('ok') and d.get('p') is not None}
        AO = J(c['asr_orders']); HO = J(c['hyb_orders'])
        T = {m: {} for m in ['asreview', 'hybrid', 'hybrid_seed0', 'jev', 'jev_file_order', 'bge', 'bge_file_order']}
        k95chk = 0; rounding = []
        for k in revs:
            rs = by[k]; y = np.array([int(r['label']) for r in rs]); N = len(y); n1 = int(y.sum()); p = np.array([float(P[r['id']]) for r in rs])
            S = J(f"{SYN}/scores_zs/{c['zs']}_{k}.json"); bge = np.array(S['bge'], float); assert len(bge) == N
            sd = {d['seed']: d for d in stored if d['review'] == k and d['method'] == 'asr_prior'}
            va = []; vh = []; vj = []; vb = []
            for s in SEEDS:
                t, k95, need = tnr95(AO[f'{k}|asr_prior|{s}'], y); va.append(t)
                if k95 != int(sd[s]['k95']): k95chk += 1   # re-run order differs from the stored run (known for a few tasks, Lancet AN-0001-08)
                t, _, _ = tnr95(HO[f'{k}|asr_jevblend_3|{s}'], y); vh.append(t)
                t, _, _ = tnr95(np.lexsort((np.random.default_rng(s).random(N), -p)), y); vj.append(t)
                t, _, _ = tnr95(np.lexsort((np.random.default_rng(s).random(N), -bge)), y); vb.append(t)
            T['asreview'][k] = mean_f(va); T['hybrid'][k] = mean_f(vh); T['hybrid_seed0'][k] = vh[0]; T['jev'][k] = mean_f(vj); T['bge'][k] = mean_f(vb)
            T['jev_file_order'][k] = tnr95(np.argsort(-p, kind='stable'), y)[0]; T['bge_file_order'][k] = tnr95(np.argsort(-bge, kind='stable'), y)[0]
            rounding.append({'review': k, 'n1': n1, 'ceil_095_n1': need, 'all_included_required': need == n1})
            rows.append({'collection': coll, 'review': k, 'N': N, 'n1': n1, 'n0': N - n1, 'ceil_0.95_n1': need, **{f'tnr95_{m}': float(T[m][k]) for m in T},
                         'tnr95_hybrid_seed_min': float(min(vh)), 'tnr95_hybrid_seed_max': float(max(vh)), 'tnr95_asreview_seed_min': float(min(va)), 'tnr95_asreview_seed_max': float(max(va))})
        R = {'n_reviews': len(revs), 'macro_tnr95': {m: float(mean_f(T[m][k] for k in revs)) for m in T},
             'macro_tnr95_ci95_bootstrap': {m: boot_ci([float(T[m][k]) for k in revs]) for m in T},
             'asreview_rerun_orders_k95_differing_from_stored_tasks': k95chk,
             'k95_rounding': {'reviews_where_95pct_recall_requires_all_included': int(sum(r['all_included_required'] for r in rounding)),
                              'n1_of_those_reviews': sorted(r['n1'] for r in rounding if r['all_included_required']),
                              'rule': 'ceil(0.95 n) = n for every n <= 19 and < n for n >= 20; so 95% recall means all included records in reviews with at most 19 included records'},
             'comparisons': {}}
        L = lambda m: [T[m][k] for k in revs]
        for a, b, nm in [('hybrid', 'asreview', 'hybrid (10-seed mean) minus ASReview'), ('jev', 'asreview', 'Jev ranking minus ASReview'), ('hybrid', 'jev', 'hybrid (10-seed mean) minus Jev ranking'),
                         ('jev', 'bge', 'Jev ranking minus bge-base (both ten random tie orders)'), ('hybrid_seed0', 'asreview', 'hybrid stored run (seed 0) minus ASReview'),
                         ('hybrid_seed0', 'jev', 'hybrid stored run (seed 0) minus Jev ranking'), ('jev_file_order', 'bge_file_order', 'Jev minus bge-base, data-file order of ties (as clef_eval.py)')]:
            R['comparisons'][f'{a}_minus_{b}'] = paired(L(a), L(b), nm)
        res['collections'][coll] = R
        log(' macro', {m: round(v, 4) for m, v in R['macro_tnr95'].items()}); log(' k95 rounding', R['k95_rounding']['reviews_where_95pct_recall_requires_all_included'], 'of', len(revs))
        for kk, v in R['comparisons'].items(): log(f"  {kk}: {v['mean_diff']:.4f} [{v['ci95_bootstrap'][0]:.4f}, {v['ci95_bootstrap'][1]:.4f}] {v['higher']}/{v['lower']}/{v['tied']} p={v['wilcoxon_p']} ({v['wilcoxon_method']})")
    # cross-checks against stored values
    j02 = J(LAN + '/AN-0001-02/results.json')['collections']
    cr = J(SYN + '/clef_results.json')['zeroshot_label']
    res['cross_checks'] = {
        'heldout_jev_random_ties_vs_lancet_an02': {'stored': j02['heldout_final']['jevonly_macro_random_ties_mean_of_10_seeds']['tnr95'], 'recomputed': res['collections']['heldout']['macro_tnr95']['jev']},
        'clef_jev_random_ties_vs_lancet_an02': {'stored': j02['clef_final']['jevonly_macro_random_ties_mean_of_10_seeds']['tnr95'], 'recomputed': res['collections']['clef']['macro_tnr95']['jev']},
        'clef_jev_file_order_vs_clef_results': {'stored_4dp': cr['jev']['TNR@95'], 'recomputed': res['collections']['clef']['macro_tnr95']['jev_file_order']},
        'clef_bge_file_order_vs_clef_results': {'stored_4dp': cr['bge']['TNR@95'], 'recomputed': res['collections']['clef']['macro_tnr95']['bge_file_order']}}
    for kk, v in res['cross_checks'].items():
        s = v.get('stored', v.get('stored_4dp')); v['match'] = bool(abs(round(v['recomputed'], 4) - round(s, 4)) < 1e-12)
    log('cross checks', res['cross_checks'])
    pd.DataFrame(rows).to_csv(os.path.join(HERE, 'per_review_tnr95.csv'), index=False)
    res['finished'] = now(); res['elapsed_seconds'] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'), indent=1, default=float)
    log('finished', res['finished'])

if __name__ == '__main__':
    main()
