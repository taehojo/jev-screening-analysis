#!/usr/bin/env python
# npj AN-0001-03 (post hoc): run-to-run variation of the hybrid (asr_jevblend_3, lambda = 3) over ten tie-breaking seeds of the
# Jev starting order, and paired comparisons repeated with the ten-seed mean. C3 remains judged on the stored single run (I-4).
# Reads (read only): an03_hybrid_{heldout,clef}.json and *_orders.json (this folder), synergy/asr_test.json and asr_clef.json
# (stored ASReview asr_prior seeds 0-9 and hybrid seed 0), record and Jev score files, Lancet AN-0001-02 jev_only_per_seed.csv,
# Lancet AN-0001-08 per_order_stopping.csv and an08_asr_*_orders.json (checks). Writes only into this folder.
import os, sys, json, math, time, subprocess, platform, warnings
from fractions import Fraction as F
import numpy as np, pandas as pd, scipy
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from an08_stopping_copy import analyse_order, wilson, clopper   # unchanged copy of Lancet AN-0001-08 an08_stopping.py (md5 in copied_scripts_md5.txt)

ROOT = '/N/project/AiLab/jev'; SYN = ROOT + '/synergy'
LAN = ROOT + '/review_pipeline/rounds/round_0001/revision/analysis'
SEEDS = list(range(10)); B = 5000; BOOT_SEED = 20260928; TOL = 1e-9
MET = ['wss95', 'wss100', 'rec@10', 'rec@20', 'rec@30']
def now(): return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S %Z'], capture_output=True, text=True).stdout.strip()
def J(p): return json.load(open(p))
def log(*a): print(*a, flush=True)
COLL = {'heldout': dict(records=SYN + '/test_records.json', jev=SYN + '/jev_test.json', stored=SYN + '/asr_test.json', new='an03_hybrid_heldout.json',
                        orders='an03_hybrid_heldout_orders.json', lan_orders=LAN + '/AN-0001-08/an08_asr_heldout_orders.json', an02='heldout_final'),
        'clef': dict(records=ROOT + '/clef/clef_records.json', jev=SYN + '/jev_clef.json', stored=SYN + '/asr_clef.json', new='an03_hybrid_clef.json',
                     orders='an03_hybrid_clef_orders.json', lan_orders=LAN + '/AN-0001-08/an08_asr_clef_orders.json', an02='clef_final')}

def exact_metrics(k95, k100, N, n1, rec_counts):
    out = {'wss95': 1 - F(k95, N) - F(1, 20), 'wss100': 1 - F(k100, N)}
    for c, v in rec_counts.items(): out[c] = F(v, n1)
    return out

def order_ints(seq, y):
    y = np.asarray(y); N = len(y); n1 = int(y.sum()); pos = np.where(y[np.asarray(seq)] == 1)[0] + 1
    k95 = int(pos[math.ceil(0.95 * n1) - 1]); k100 = int(pos[n1 - 1])
    rc = {f'rec@{int(f*100)}': int((pos <= math.floor(f * N)).sum()) for f in (0.1, 0.2, 0.3)}
    return k95, k100, rc

def mean_f(v): v = list(v); return sum(v, F(0)) / len(v)

def boot_ci(d, seed=BOOT_SEED):
    d = np.asarray(d, float); rng = np.random.default_rng(seed); m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

def method_rule(d):
    nz = d[d != 0]; ties = len(nz) - len(np.unique(np.abs(nz))); zeros = int((d == 0).sum())
    if zeros == 0 and ties == 0 and len(d) <= 50: return 'exact'
    if len(d) <= 13: return 'exact (permutation; zero or tied differences)'
    return 'asymptotic'

def paired(a_ex, b_ex, label):
    d = [x - y for x, y in zip(a_ex, b_ex)]; dfl = np.array([float(v) for v in d])
    hi = sum(v > 0 for v in d); lo = sum(v < 0 for v in d); ze = sum(v == 0 for v in d)
    out = {'label': label, 'n': len(d), 'mean_a': float(mean_f(a_ex)), 'mean_b': float(mean_f(b_ex)), 'mean_diff': float(mean_f(d)),
           'ci95_bootstrap': boot_ci(dfl), 'higher': int(hi), 'lower': int(lo), 'tied': int(ze), 'wilcoxon_p': None, 'wilcoxon_method': None}
    if np.any(dfl != 0):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); out['wilcoxon_p'] = float(wilcoxon(dfl).pvalue)
        out['wilcoxon_method'] = method_rule(dfl)
    return out

def main():
    t0 = time.time(); res = {'analysis_id': 'AN-0001-03 (npj round 1)', 'post_hoc': True, 'started': now(),
                             'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__},
                             'settings': {'hybrid': 'asr_jevblend_3 (lambda = 3), tie-breaking seeds 0-9 of the Jev starting order (np.random.RandomState(seed).rand in lexsort)',
                                          'asreview': 'asr_prior seeds 0-9 as stored in synergy/asr_test.json and asr_clef.json', 'jev_only': 'np.lexsort((default_rng(seed).random(N), -p)), seeds 0-9 (as Lancet AN-0001-02)',
                                          'per_review_value': 'mean over the ten seeds (exact fractions from k95, k100 and counts)', 'bootstrap': f'percentile, {B} resamples of reviews, seed {BOOT_SEED}',
                                          'wilcoxon': 'scipy.stats.wilcoxon defaults on exact differences converted to float; method from the SciPy auto rule',
                                          'stopping': 'knee (150-record minimum) and statistical criterion (Callaghan and Muller-Hansen 2020) with analyse_order() of Lancet AN-0001-08'},
                             'collections': {}}
    per_rows = []; seed_rows = []
    for coll, c in COLL.items():
        log('==', coll)
        stored = J(c['stored']); new = J(os.path.join(HERE, c['new'])); orders = J(os.path.join(HERE, c['orders']))
        revs = sorted({d['review'] for d in stored if d['method'] == 'asr_jevblend_3' and d.get('wss95') is not None})
        by = {}
        for r in J(c['records']): by.setdefault(r['review'], []).append(r)
        P = {d['id']: d['p'] for d in J(c['jev']) if d.get('ok') and d.get('p') is not None}
        # ---- reproduction checks
        keys = ['wss95', 'wss100', 'rec@10', 'rec@20', 'rec@30', 'k95', 'k100', 'knee_work', 'knee_rec', 'target_work', 'target_rec']
        st0 = {d['review']: d for d in stored if d['method'] == 'asr_jevblend_3'}
        nw = {(d['review'], d['seed']): d for d in new if d['method'] == 'asr_jevblend_3'}
        missing = [(k, s) for k in revs for s in SEEDS if (k, s) not in nw or nw[(k, s)].get('wss95') is None]
        rep = {'n_tasks_expected': len(revs) * 10, 'n_tasks_found': len(revs) * 10 - len(missing), 'missing': missing, 'seed0_diffs': []}
        for k in revs:
            for kk in keys:
                if abs(float(nw[(k, 0)][kk]) - float(st0[k][kk])) > TOL: rep['seed0_diffs'].append({'review': k, 'metric': kk, 'new': nw[(k, 0)][kk], 'stored': st0[k][kk]})
        lo = J(c['lan_orders']); rep['seed0_order_identical_to_lancet_an08'] = sum(1 for k in revs if lo.get(f'{k}|asr_jevblend_3|0') == orders.get(f'{k}|asr_jevblend_3|0'))
        rep['seed0_reproduces_stored'] = len(rep['seed0_diffs']) == 0
        log('reproduction', {x: rep[x] for x in ['n_tasks_found', 'seed0_reproduces_stored', 'seed0_order_identical_to_lancet_an08']})
        # ASReview stored, per review exact seed mean
        asr = {}
        for d in stored:
            if d['method'] == 'asr_prior' and d.get('wss95') is not None:
                N, n1 = int(d['N']), int(d['n1'])
                rc = {m: int(round(d[m] * n1)) for m in ['rec@10', 'rec@20', 'rec@30']}
                asr.setdefault(d['review'], []).append(exact_metrics(int(d['k95']), int(d['k100']), N, n1, rc))
        assert all(len(asr[k]) == 10 for k in revs)
        # stored stopping values for checks and comparators
        pos_ = pd.read_csv(LAN + '/AN-0001-08/per_order_stopping.csv'); pos_ = pos_[pos_.collection == coll]
        # Jev-only per seed (recomputed) and check against Lancet AN-0001-02
        j02 = pd.read_csv(LAN + '/AN-0001-02/jev_only_per_seed.csv'); j02 = j02[j02.collection == c['an02']].set_index(['review', 'seed'])
        jev_check = 0; hyb_stat_check = []
        H = {}; JO = {}; HS = {}
        for k in revs:
            rs = by[k]; y = np.array([int(r['label']) for r in rs]); N = len(y); n1 = int(y.sum()); p = np.array([float(P[r['id']]) for r in rs])
            JO[k] = []
            for s in SEEDS:
                seq = np.lexsort((np.random.default_rng(s).random(N), -p)); k95, k100, rc = order_ints(seq, y)
                JO[k].append(exact_metrics(k95, k100, N, n1, rc))
                if int(j02.loc[(k, s), 'k95']) != k95 or int(j02.loc[(k, s), 'k100']) != k100: jev_check += 1
            H[k] = []; HS[k] = []
            for s in SEEDS:
                seq = orders[f'{k}|asr_jevblend_3|{s}']; k95, k100, rc = order_ints(seq, y)
                if k95 != int(nw[(k, s)]['k95']) or k100 != int(nw[(k, s)]['k100']): raise SystemExit(f'STOP: order and metrics disagree {k} {s}')
                H[k].append(exact_metrics(k95, k100, N, n1, rc))
                a = analyse_order(seq, y, N)
                HS[k].append(a)
                seed_rows.append({'collection': coll, 'review': k, 'seed': s, 'N': N, 'n1': n1, 'k95': k95, 'k100': k100, **{m: float(H[k][-1][m]) for m in MET},
                                  'knee_k': a['knee_k'], 'knee_work': a['knee_work'], 'knee_rec': a['knee_rec'], 'stat_k': a['stat_k'], 'stat_work': a['stat_work'], 'stat_rec': a['stat_rec'],
                                  'target_work': nw[(k, s)]['target_work'], 'target_rec': nw[(k, s)]['target_rec'], 'atd': a['atd']})
                if abs(a['knee_work'] - nw[(k, s)]['knee_work']) > TOL: raise SystemExit(f'STOP: knee differs {k} {s}')
                if s == 0:
                    row = pos_[(pos_.review == k) & (pos_.method == 'asr_jevblend_3') & (pos_.seed == 0)].iloc[0]
                    if int(row.stat_k) != a['stat_k'] or int(row.knee_k) != a['knee_k']: hyb_stat_check.append(k)
        rep['jev_only_k_mismatches_vs_lancet_an02'] = jev_check; rep['hybrid_seed0_stopping_mismatches_vs_lancet_an08'] = hyb_stat_check
        # ---- per-review ten-seed means
        Hm = {k: {m: mean_f(x[m] for x in H[k]) for m in MET} for k in revs}
        H0 = {k: H[k][0] for k in revs}
        Am = {k: {m: mean_f(x[m] for x in asr[k]) for m in MET} for k in revs}
        Jm = {k: {m: mean_f(x[m] for x in JO[k]) for m in MET} for k in revs}
        R = {'n_reviews': len(revs), 'reproduction': rep}
        R['macro'] = {'hybrid_10seed_mean': {m: float(mean_f(Hm[k][m] for k in revs)) for m in MET}, 'hybrid_seed0_stored_run': {m: float(mean_f(H0[k][m] for k in revs)) for m in MET},
                      'asreview_10seed_mean': {m: float(mean_f(Am[k][m] for k in revs)) for m in MET}, 'jev_only_10seed_mean': {m: float(mean_f(Jm[k][m] for k in revs)) for m in MET}}
        per_seed_macro = {m: [float(mean_f(H[k][s][m] for k in revs)) for s in SEEDS] for m in MET}
        R['hybrid_macro_by_seed'] = per_seed_macro
        R['hybrid_macro_seed_range'] = {m: [min(v), max(v)] for m, v in per_seed_macro.items()}
        R['hybrid_macro_seed_sd'] = {m: float(np.std(v, ddof=1)) for m, v in per_seed_macro.items()}
        # within-review spread of WSS@95 over seeds
        spread = np.array([float(max(x['wss95'] for x in H[k]) - min(x['wss95'] for x in H[k])) for k in revs])
        # descriptive (added before the first run of this script; log entry of AN-0001-03): position from which the orders of all seeds coincide
        conv = {}
        for k in revs:
            seqs = [orders[f'{k}|asr_jevblend_3|{s}'] for s in SEEDS]; N = len(seqs[0]); t = N
            while t > 0 and all(sq[t - 1] == seqs[0][t - 1] for sq in seqs): t -= 1
            # t = number of leading positions in which at least one seed differs from seed 0; the prefix sets must then be equal
            same_set = all(set(sq[:t]) == set(seqs[0][:t]) for sq in seqs)
            conv[k] = {'N': N, 'positions_until_orders_coincide': t, 'proportion_of_N': t / N, 'prefix_sets_equal': same_set,
                       'n_distinct_full_orders': len({tuple(sq) for sq in seqs})}
        R['order_convergence_over_seeds'] = {'per_review': conv, 'max_positions': max(v['positions_until_orders_coincide'] for v in conv.values()),
                                             'median_positions': float(np.median([v['positions_until_orders_coincide'] for v in conv.values()])),
                                             'reviews_with_any_order_difference': sum(v['n_distinct_full_orders'] > 1 for v in conv.values()),
                                             'reviews_with_k95_or_k100_difference': sum(len({(x['wss95'], x['wss100']) for x in H[k]}) > 1 for k in revs)}
        R['hybrid_within_review_wss95_range'] = {'median': float(np.median(spread)), 'max': float(spread.max()), 'review_max': revs[int(spread.argmax())], 'n_reviews_no_variation': int((spread == 0).sum())}
        # ---- paired comparisons
        cmp_ = {}
        for m in MET:
            cmp_[f'hybrid10_minus_asreview.{m}'] = paired([Hm[k][m] for k in revs], [Am[k][m] for k in revs], f'hybrid (10-seed mean) minus ASReview: {m}')
            cmp_[f'hybrid10_minus_jevonly.{m}'] = paired([Hm[k][m] for k in revs], [Jm[k][m] for k in revs], f'hybrid (10-seed mean) minus Jev ranking (10 tie seeds): {m}')
            cmp_[f'hybrid0_minus_asreview.{m}'] = paired([H0[k][m] for k in revs], [Am[k][m] for k in revs], f'hybrid stored run (seed 0) minus ASReview: {m} (reproduction of the stored comparison)')
            cmp_[f'hybrid0_minus_jevonly.{m}'] = paired([H0[k][m] for k in revs], [Jm[k][m] for k in revs], f'hybrid stored run (seed 0) minus Jev ranking: {m} (reproduction)')
            cmp_[f'jevonly_minus_asreview.{m}'] = paired([Jm[k][m] for k in revs], [Am[k][m] for k in revs], f'Jev ranking minus ASReview: {m}')
        R['comparisons'] = cmp_
        # each seed separately: hybrid seed s minus Jev ranking and minus ASReview (WSS@95, WSS@100); all seeds reported
        R['per_seed_comparisons'] = {}
        for m in ['wss95', 'wss100']:
            for tag, comp in [('asreview', Am), ('jevonly', Jm)]:
                R['per_seed_comparisons'][f'hybrid_seed_minus_{tag}.{m}'] = [dict(seed=s, **{kk: v for kk, v in paired([H[k][s][m] for k in revs], [comp[k][m] for k in revs], '').items() if kk != 'label'}) for s in SEEDS]
        # C3 numeric rule (difference >= 0.05 and Wilcoxon p < 0.05) applied to the ten-seed mean as a sensitivity check
        c3 = cmp_['hybrid10_minus_asreview.wss95']
        R['c3_rule_on_10seed_mean_sensitivity'] = {'difference': c3['mean_diff'], 'p': c3['wilcoxon_p'], 'meets_numeric_rule': bool(c3['mean_diff'] >= 0.05 and c3['wilcoxon_p'] < 0.05),
                                                   'seeds_meeting_rule': sum(1 for x in R['per_seed_comparisons']['hybrid_seed_minus_asreview.wss95'] if x['mean_diff'] >= 0.05 and x['wilcoxon_p'] < 0.05)}
        # ---- knee and statistical criterion on the hybrid ranking
        st = {}
        for rule in ['knee', 'stat']:
            rec_mean = np.array([np.mean([HS[k][s][f'{rule}_rec'] for s in SEEDS]) for k in revs]); work_mean = np.array([np.mean([HS[k][s][f'{rule}_work'] for s in SEEDS]) for k in revs])
            Ns = np.array([len(by[k]) for k in revs]); kk = int((rec_mean >= 0.95).sum())
            per_seed_rel = [int(sum(HS[k][s][f'{rule}_rec'] >= 0.95 for k in revs)) for s in SEEDS]; per_seed_work = [float(np.mean([HS[k][s][f'{rule}_work'] for k in revs])) for s in SEEDS]
            rec0 = np.array([HS[k][0][f'{rule}_rec'] for k in revs]); work0 = np.array([HS[k][0][f'{rule}_work'] for k in revs])
            st[rule] = {'reliable_reviews_seed_mean_recall': kk, 'reliability': kk / len(revs), 'wilson95': wilson(kk, len(revs)), 'clopper_pearson95': clopper(kk, len(revs)),
                        'macro_work_10seed_mean': float(work_mean.mean()), 'pooled_work_10seed_mean': float((work_mean * Ns).sum() / Ns.sum()), 'mean_recall': float(rec_mean.mean()), 'min_recall': float(rec_mean.min()),
                        'reliable_reviews_by_seed': per_seed_rel, 'reliable_reviews_seed_range': [min(per_seed_rel), max(per_seed_rel)],
                        'macro_work_by_seed': per_seed_work, 'macro_work_seed_range': [min(per_seed_work), max(per_seed_work)],
                        'seed0_reliable_reviews': int((rec0 >= 0.95).sum()), 'seed0_macro_work': float(work0.mean()), 'failing_reviews_seed_mean': [k for k, v in zip(revs, rec_mean) if v < 0.95]}
        R['stopping_on_hybrid_ranking'] = st
        res['collections'][coll] = R
        for kk in ['hybrid10_minus_asreview.wss95', 'hybrid10_minus_jevonly.wss95', 'hybrid0_minus_jevonly.wss95', 'hybrid10_minus_asreview.wss100', 'hybrid10_minus_jevonly.wss100']:
            v = cmp_[kk]; log(f"  {kk}: {v['mean_diff']:.4f} [{v['ci95_bootstrap'][0]:.4f}, {v['ci95_bootstrap'][1]:.4f}] {v['higher']}/{v['lower']}/{v['tied']} p={v['wilcoxon_p']} ({v['wilcoxon_method']})")
        log('  macro', {kk: {m: round(v, 4) for m, v in d.items()} for kk, d in R['macro'].items()})
        log('  seed range', R['hybrid_macro_seed_range']); log('  stopping', {r: (v['reliable_reviews_seed_mean_recall'], round(v['macro_work_10seed_mean'], 4), v['reliable_reviews_seed_range']) for r, v in st.items()})
        per_rows += [{'collection': coll, 'review': k, 'N': len(by[k]), **{f'hybrid10_{m}': float(Hm[k][m]) for m in MET}, **{f'hybrid0_{m}': float(H0[k][m]) for m in MET},
                      **{f'asr10_{m}': float(Am[k][m]) for m in MET}, **{f'jevonly10_{m}': float(Jm[k][m]) for m in MET},
                      'hybrid_wss95_seed_min': float(min(x['wss95'] for x in H[k])), 'hybrid_wss95_seed_max': float(max(x['wss95'] for x in H[k]))} for k in revs]
    pd.DataFrame(per_rows).to_csv(os.path.join(HERE, 'per_review.csv'), index=False)
    pd.DataFrame(seed_rows).to_csv(os.path.join(HERE, 'per_seed_hybrid.csv'), index=False)
    res['finished'] = now(); res['elapsed_seconds'] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'), indent=1, default=float)
    log('finished', res['finished'], res['elapsed_seconds'], 's')

if __name__ == '__main__':
    main()
