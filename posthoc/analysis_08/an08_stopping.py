#!/usr/bin/env python
# AN-0001-08: statistical stopping criterion (Callaghan and Muller-Hansen 2020) as a post-hoc comparator, combined
# workflow (label-free Jev ranking followed by a labelled statistical stop), average time to discovery, recall curves and
# leave-one-review-out influence. Post hoc; C4 is unchanged (INTEGRITY_RULES I-4).
# Inputs (read only): re-run simulations with saved labelling orders (an08_asr_{heldout,clef}.json, *_orders.json), the stored
# originals (synergy/asr_test.json, asr_clef.json), record and Jev score files, AN-0001-02, AN-0001-04 and AN-0001-13 per-review files.
import os, sys, json, time, math, subprocess
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import hypergeom, beta, wilcoxon
from scipy.special import gammaln
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from al_sim import metrics, knee_stop   # unchanged copy of synergy/al_sim.py

ROOT = '/N/project/AiLab/jev'; HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(ROOT, 'review_pipeline/rounds/round_0001/revision/analysis')
TAU = 0.07; RT = 0.95; CONF = 0.95; ALPHA = 1 - CONF; SEEDS = list(range(10)); BOOT_SEED = 1008; B = 5000
GRID = np.round(np.linspace(0, 1, 201), 3)
COLL = {
    'heldout': dict(records='synergy/test_records.json', jev='synergy/jev_test.json', stored='synergy/asr_test.json',
                    new='an08_asr_heldout.json', orders='an08_asr_heldout_orders.json', an04='AN-0001-04/per_review_heldout.csv', an02='heldout_final'),
    'clef': dict(records='clef/clef_records.json', jev='synergy/jev_clef.json', stored='synergy/asr_clef.json',
                 new='an08_asr_clef.json', orders='an08_asr_clef_orders.json', an04='AN-0001-04/per_review_clef.csv', an02='clef_final'),
}
def now(): return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S %Z'], capture_output=True, text=True).stdout.strip()

# ---------- statistical stopping criterion (paper eqs 5 and 10, ranked quasi-sampling variant; buscarpy calculate_h0 equivalent)
_LF = None
def _lnfact_table(n):
    global _LF
    if _LF is None or len(_LF) < n + 2: _LF = gammaln(np.arange(0, n + 2, dtype=float) + 1.0)   # _LF[m] = ln m!
    return _LF
def _lnC(a, b):
    """ln C(a, b) elementwise for integer arrays; -inf where the coefficient is zero (b < 0 or b > a)."""
    ok = (b >= 0) & (b <= a) & (a >= 0); aa = np.clip(a, 0, None); bb = np.clip(b, 0, None); ab = np.clip(aa - bb, 0, None)
    v = _LF[aa] - _LF[bb] - _LF[ab]; return np.where(ok, v, -np.inf)

def h0_pvalue(lab, N, rt=RT):
    """lab: 0/1 labels in screening order (the records screened so far). Null hypothesis: recall < rt.
    For every look-back window (the last i screened records treated as a sample from the i + unseen records), Ktar is the smallest
    number of relevant records in that population compatible with H0 and p = P(X <= k), X ~ Hypergeometric(population, Ktar, i)
    (paper eqs 5 and 10; buscarpy.calculate_h0). Because Ktar = k + c with c = floor(r_seen/rt + 1) - r_seen the same for every
    window, P(X <= k) equals P(Y >= c) where Y is the number of relevant records among the N - t unseen records, and is computed as
    1 - sum_{x<c} pmf_Y(x) with log-binomial coefficients from a factorial table (exact, and fast for large reviews).
    Returns (min p over windows, number of windows where H0 is impossible (Ktar > population; p = 0 there), 0)."""
    lab = np.asarray(lab, np.int64); t = len(lab); r_seen = int(lab.sum()); D = N - t
    c = int(math.floor(r_seen / rt + 1)) - r_seen                       # same floor expression as buscarpy.calculate_h0
    cum = np.cumsum(lab[::-1]); sizes = np.arange(1, t + 1); M = D + sizes; K = cum + c
    impossible = K > M
    if D <= 0: return 0.0, int(impossible.sum()), 0                      # nothing unseen: recall = 1 >= rt, H0 rejected
    _lnfact_table(N)
    lnCMD = _lnC(M, np.full(t, D)); S = np.zeros(t)
    for x in range(0, min(c, D + 1)):                                     # x = 0 .. c-1 (terms with x > D vanish)
        S += np.exp(_lnC(K, np.full(t, x)) + _lnC(M - K, np.full(t, D - x)) - lnCMD)
    p = np.clip(1.0 - S, 0.0, 1.0); p[impossible] = 0.0
    return float(p.min()), int(impossible.sum()), 0

def schedule(N, extra=()):
    step = 1 if N <= 2000 else max(1, N // 2000)
    s = set(range(step, N + 1, step)); s.add(N); s.update(int(e) for e in extra if 1 <= e <= N)
    return sorted(s), step

def wilson(k, n, z=1.959963984540054):
    ph = k / n; d = 1 + z * z / n; c = ph + z * z / (2 * n); h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]
def clopper(k, n, a=0.05):
    return [0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))]

def analyse_order(seq, y, N, extra_points=(), keep_p=False):
    seq = np.asarray(seq); ys = y[seq]; cum = np.cumsum(ys); n1 = int(y.sum())
    m = metrics(seq, y); ks = knee_stop(seq, y)
    out = {'k95': m['k95'], 'k100': m['k100'], 'wss95': m['wss95'], 'wss100': m['wss100'], 'rec@10': m['rec@10'], 'rec@20': m['rec@20'], 'rec@30': m['rec@30'],
           'knee_k': int(ks), 'knee_work': ks / N, 'knee_rec': float(cum[ks - 1] / n1)}
    sched, step = schedule(N, extra_points)
    ps = np.empty(len(sched)); imp = 0; nn = 0
    for j, t in enumerate(sched):
        ps[j], a, b = h0_pvalue(ys[:t], N); imp += a; nn += b
    trig = np.where(ps < ALPHA)[0]
    if len(trig): k_stat = sched[trig[0]]; out['stat_triggered'] = True
    else: k_stat = N; out['stat_triggered'] = False
    out.update({'stat_k': int(k_stat), 'stat_work': k_stat / N, 'stat_rec': float(cum[k_stat - 1] / n1), 'stat_step': step, 'stat_n_eval': len(sched),
                'stat_h0_impossible_windows': imp, 'stat_nan': nn, 'stat_p_at_end': float(ps[-1]), 'stat_p_at_k95': None})
    # p at the first point >= k95 (how far the criterion lags the true 95% point)
    j95 = int(np.searchsorted(sched, m['k95']))
    if j95 < len(sched): out['stat_p_at_k95'] = float(ps[j95])
    # combined: read at least to the extra point (threshold), then continue until the criterion is met
    for e in extra_points:
        e = int(e); je = int(np.searchsorted(sched, max(e, 1)))
        tr = [t for t, p_ in zip(sched[je:], ps[je:]) if p_ < ALPHA]
        kk = tr[0] if tr else N
        out.update({'comb_k': int(kk), 'comb_work': kk / N, 'comb_rec': float(cum[kk - 1] / n1), 'comb_triggered': bool(tr)})
    td = np.where(ys == 1)[0] + 1
    out['atd'] = float(td.mean() / N); out['td_positions'] = td.tolist()
    reads = np.floor(GRID * N).astype(int); rc = np.where(reads >= 1, cum[np.maximum(reads, 1) - 1] / n1, 0.0)
    out['recall_curve'] = rc.tolist()
    if keep_p: out['p_curve'] = {'schedule': [int(t) for t in sched], 'p': ps.tolist(), 'labels_in_order': ys.astype(int).tolist()}
    return out

def task(args):
    coll, review, y, p, orders = args
    t0 = time.time(); N = len(y); y = np.asarray(y); p = np.asarray(p, float)
    t_thr = int((p >= TAU).sum())
    res = {'collection': coll, 'review': review, 'N': N, 'n1': int(y.sum()), 'thr_k': t_thr, 'thr_work': t_thr / N, 'thr_rec': float(y[p >= TAU].sum() / y.sum()), 'orders': {}}
    for key, seq in orders.items():
        res['orders'][key] = analyse_order(seq, y, N, keep_p=(key == 'asr_jevblend_3|0'))
    for s in SEEDS:
        rng = np.random.default_rng(s); seq = np.lexsort((rng.random(N), -p))
        r = analyse_order(seq, y, N, extra_points=(t_thr,)); res['orders'][f'jev_only|{s}'] = r
    res['secs'] = time.time() - t0
    return res

def paired(x, y, label, seed=BOOT_SEED):
    d = np.asarray(x, float) - np.asarray(y, float); n = len(d)
    rng = np.random.default_rng(seed); idx = rng.integers(0, n, (B, n)); mm = d[idx].mean(1)
    out = {'label': label, 'n': n, 'mean_diff': float(d.mean()), 'ci95': [float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))],
           'wins': int((d > 0).sum()), 'losses': int((d < 0).sum()), 'ties': int((d == 0).sum()), 'wilcoxon_p': None}
    if n > 5 and np.any(d != 0): out['wilcoxon_p'] = float(wilcoxon(np.asarray(x, float), np.asarray(y, float)).pvalue)
    return out
def boot_mean(v, seed=BOOT_SEED):
    v = np.asarray(v, float); rng = np.random.default_rng(seed); idx = rng.integers(0, len(v), (B, len(v))); mm = v[idx].mean(1)
    return [float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))]

def rule_summary(T, prefix, N, label):
    """T: per-review DataFrame with columns f'{prefix}_work', f'{prefix}_rec' (seed means) and optionally f'{prefix}_rel_perseed'."""
    work = T[f'{prefix}_work'].values; rec = T[f'{prefix}_rec'].values; n = len(T); k = int((rec >= 0.95).sum())
    out = {'label': label, 'n_reviews': n, 'reliable_reviews': k, 'reliability': k / n, 'wilson95': wilson(k, n), 'clopper_pearson95': clopper(k, n),
           'macro_work': float(work.mean()), 'macro_work_boot95': boot_mean(work), 'median_work': float(np.median(work)), 'work_iqr': [float(np.percentile(work, 25)), float(np.percentile(work, 75))],
           'work_range': [float(work.min()), float(work.max())], 'pooled_work': float((work * N).sum() / N.sum()),
           'mean_recall': float(rec.mean()), 'min_recall': float(rec.min()), 'median_recall': float(np.median(rec)),
           'failing_reviews': T.review[rec < 0.95].tolist()}
    if f'{prefix}_rel_perseed' in T: out['reliability_per_seed_mean'] = float(T[f'{prefix}_rel_perseed'].mean())
    return out

def loo_ranges(revs, fn):
    """fn(mask) -> dict of scalar values; returns for every key the full value, min, max and the review whose removal gives them."""
    full = fn(np.ones(len(revs), bool)); vals = {k: [] for k in full}
    for i in range(len(revs)):
        m = np.ones(len(revs), bool); m[i] = False; v = fn(m)
        for k in full: vals[k].append(v[k])
    out = {}
    for k in full:
        a = np.array(vals[k], float);
        if np.all(np.isnan(a)): out[k] = {'full': full[k]}; continue
        out[k] = {'full': full[k], 'loo_min': float(np.nanmin(a)), 'loo_max': float(np.nanmax(a)), 'review_min': revs[int(np.nanargmin(a))], 'review_max': revs[int(np.nanargmax(a))],
                  'loo_values': {r: float(x) for r, x in zip(revs, a)}}
    return out

def main():
    t_start = time.time(); started = now(); print('AN-0001-08 started', started, flush=True)
    res = {'analysis_id': 'AN-0001-08', 'started': started, 'settings': {
        'statistical_criterion': 'Callaghan and Muller-Hansen 2020 (Syst Rev 9:273), ranked quasi-sampling variant: H0 recall < 0.95; for every look-back window Ktar = floor(r_seen/0.95 + 1 - r_before) and p = P(X <= k), X ~ Hypergeometric(i + unseen, Ktar, i); stop when min p < 0.05 (confidence 0.95)',
        'evaluation_points': 'after every screened record when N <= 2000, else every max(1, N//2000) records (plus N and, for Jev-only orders, the threshold point)',
        'h0_impossible': 'windows where Ktar exceeds the population are given p = 0 (H0 cannot hold); their number is recorded',
        'tau': TAU, 'recall_target': RT, 'confidence': CONF, 'seeds': SEEDS, 'bootstrap': {'resamples': B, 'seed': BOOT_SEED},
        'reliability': 'primary: seed-mean recall at stop >= 0.95 per review (as final_eval.py); secondary: per-seed proportion',
        'atd': 'Ferdinands et al 2023: time to discovery of an included record = number of records screened when it is found; ATD = mean over included records (averaged over seeds first for ASReview and Jev-only), divided by N',
        'jev_only_order': 'np.lexsort((default_rng(seed).random(N), -p)), seeds 0-9 (as AN-0001-02)'}, 'collections': {}}
    per_review_rows = {}; per_order_rows = []; curves_macro = {}; curves_review = []; p_curves = {}
    repro_rows = []
    for coll, c in COLL.items():
        R = json.load(open(os.path.join(ROOT, c['records']))); J = {d['id']: d['p'] for d in json.load(open(os.path.join(ROOT, c['jev']))) if d.get('ok') and d.get('p') is not None}
        by = {}
        for r in R: by.setdefault(r['review'], []).append(r)
        new = pd.DataFrame(json.load(open(os.path.join(HERE, c['new'])))); stored = pd.DataFrame(json.load(open(os.path.join(ROOT, c['stored']))))
        orders = json.load(open(os.path.join(HERE, c['orders'])))
        # ---- reproduction check of the re-run against the stored values
        keys = ['wss95', 'wss100', 'rec@10', 'rec@20', 'rec@30', 'k95', 'k100', 'knee_work', 'knee_rec', 'target_work', 'target_rec', 'N', 'n1']
        S = stored.set_index(['review', 'method', 'seed']); Nw = new.set_index(['review', 'method', 'seed'])
        common = Nw.index.intersection(S.index); missing = Nw.index.difference(S.index)
        mx = {}; bad = {}
        for k in keys:
            d = (Nw.loc[common, k].astype(float) - S.loc[common, k].astype(float)).abs(); mx[k] = float(d.max()); bad[k] = int((d > 1e-9).sum())
            for (rv, me, sd), dv in d[d > 1e-9].items(): repro_rows.append({'collection': coll, 'review': rv, 'method': me, 'seed': sd, 'key': k, 'new': float(Nw.loc[(rv, me, sd), k]), 'stored': float(S.loc[(rv, me, sd), k]), 'abs_diff': float(dv)})
        cres = {'n_tasks_rerun': int(len(new)), 'n_tasks_compared': int(len(common)), 'n_tasks_not_in_stored': int(len(missing)), 'reproduction_max_abs_diff': mx, 'reproduction_n_diff_gt_1e-9': bad,
                'reproduction_exact': all(v == 0 for v in bad.values())}
        print(coll, 'reproduction', cres['reproduction_exact'], mx, flush=True)
        # ---- per-review orders
        revs = sorted(set(new.review)); tasks = []
        for k in revs:
            rs = by[k]; y = np.array([r['label'] for r in rs]); p = np.array([J[r['id']] for r in rs], float)
            ords = {f'asr_prior|{s}': orders[f'{k}|asr_prior|{s}'] for s in SEEDS}; ords['asr_jevblend_3|0'] = orders[f'{k}|asr_jevblend_3|0']
            tasks.append((coll, k, y, p, ords))
        tasks.sort(key=lambda t: -len(t[2]))
        t0 = time.time()
        with Pool(24) as pool: outs = pool.map(task, tasks, chunksize=1)
        print(coll, 'stopping analysis of', len(outs), 'reviews in', f'{time.time()-t0:.0f}s', flush=True)
        # ---- consistency of the saved orders with the stored metrics (k95, k100, knee) and of the threshold rule with AN-0001-04
        A04 = pd.read_csv(os.path.join(AN, c['an04'])).set_index('review'); A02 = pd.read_csv(os.path.join(AN, 'AN-0001-02/per_review.csv')); A02 = A02[A02.collection == c['an02']].set_index('review')
        ord_bad = 0; thr_bad = 0; jo_bad = 0
        rows = []
        for o in sorted(outs, key=lambda o: o['review']):
            k = o['review']; N = o['N']; p_curves[f'{coll}|{k}'] = {'N': N, 'n1': o['n1'], **o['orders']['asr_jevblend_3|0']['p_curve']}
            for key, r in o['orders'].items():
                me, sd = key.split('|'); sd = int(sd)
                if me != 'jev_only':
                    st = Nw.loc[(k, me, sd)]   # the saved order must reproduce the metrics written by the same run
                    if abs(r['k95'] - st['k95']) > 1e-9 or abs(r['k100'] - st['k100']) > 1e-9 or abs(r['knee_work'] - st['knee_work']) > 1e-9 or abs(r['knee_rec'] - st['knee_rec']) > 1e-9: ord_bad += 1
                per_order_rows.append({'collection': coll, 'review': k, 'N': N, 'n1': o['n1'], 'method': me, 'seed': sd, **{kk: v for kk, v in r.items() if kk not in ('td_positions', 'recall_curve', 'p_curve')}})
            if abs(o['thr_work'] - A04.loc[k, 'thr_work']) > 1e-9 or abs(o['thr_rec'] - A04.loc[k, 'thr_rec']) > 1e-9: thr_bad += 1
            jo95 = np.mean([o['orders'][f'jev_only|{s}']['wss95'] for s in SEEDS])
            if abs(jo95 - A02.loc[k, 'jevonly_wss95']) > 1e-9: jo_bad += 1
            row = {'collection': coll, 'review': k, 'N': N, 'n1': o['n1'], 'prevalence': o['n1'] / N, 'thr_k': o['thr_k'], 'thr_work': o['thr_work'], 'thr_rec': o['thr_rec'], 'thr_reliable': int(o['thr_rec'] >= 0.95),
                   'stat_secs': o['secs']}
            for me_, mk in [('asr', 'asr_prior'), ('hybrid', 'asr_jevblend_3')]:   # stored (v0) seed means, used for the leave-one-out of the headline values
                ss = stored[(stored.review == k) & (stored.method == mk)]
                for q in ['wss95', 'wss100', 'knee_work', 'knee_rec']: row[f'{me_}_{q}_stored'] = float(ss[q].mean())
            for me, keys_ in [('asr', [f'asr_prior|{s}' for s in SEEDS]), ('hybrid', ['asr_jevblend_3|0']), ('jev', [f'jev_only|{s}' for s in SEEDS])]:
                rr = [o['orders'][kk] for kk in keys_]
                for q in ['knee_work', 'knee_rec', 'stat_work', 'stat_rec', 'atd', 'wss95', 'wss100', 'k95', 'k100', 'stat_k', 'knee_k']:
                    row[f'{me}_{q}'] = float(np.mean([x[q] for x in rr]))
                row[f'{me}_knee_rel_perseed'] = float(np.mean([x['knee_rec'] >= 0.95 for x in rr])); row[f'{me}_stat_rel_perseed'] = float(np.mean([x['stat_rec'] >= 0.95 for x in rr]))
                row[f'{me}_stat_triggered_frac'] = float(np.mean([x['stat_triggered'] for x in rr])); row[f'{me}_stat_h0_impossible_windows'] = int(sum(x['stat_h0_impossible_windows'] for x in rr))
                row[f'{me}_stat_nan'] = int(sum(x['stat_nan'] for x in rr)); row[f'{me}_stat_p_at_k95'] = float(np.mean([x['stat_p_at_k95'] for x in rr if x['stat_p_at_k95'] is not None])) if any(x['stat_p_at_k95'] is not None for x in rr) else None
                row[f'{me}_stat_step'] = rr[0]['stat_step']
                if me == 'jev':
                    for q in ['comb_work', 'comb_rec', 'comb_k']: row[f'{me}_{q}'] = float(np.mean([x[q] for x in rr]))
                    row['jev_comb_rel_perseed'] = float(np.mean([x['comb_rec'] >= 0.95 for x in rr]))
                curve = np.mean([x['recall_curve'] for x in rr], axis=0)
                curves_review.append({'collection': coll, 'review': k, 'method': me, **{f'x{g:.3f}': float(v) for g, v in zip(GRID, curve)}})
            rows.append(row)
        T = pd.DataFrame(rows); per_review_rows[coll] = T
        cres.update({'orders_consistent_with_stored_metrics': ord_bad == 0, 'orders_inconsistent_count': ord_bad, 'threshold_rule_matches_AN04': thr_bad == 0, 'jev_only_wss95_matches_AN02': jo_bad == 0,
                     'n_reviews': len(T), 'records_total': int(T.N.sum()), 'included_total': int(T.n1.sum()), 'stat_seconds_sum': float(T.stat_secs.sum()),
                     'stat_h0_impossible_windows_total': int(T[['asr_stat_h0_impossible_windows', 'hybrid_stat_h0_impossible_windows', 'jev_stat_h0_impossible_windows']].sum().sum()),
                     'stat_nan_total': int(T[['asr_stat_nan', 'hybrid_stat_nan', 'jev_stat_nan']].sum().sum()),
                     'reviews_with_coarse_evaluation': T.review[T.asr_stat_step > 1].tolist(), 'evaluation_steps': {r: int(s) for r, s in zip(T.review, T.asr_stat_step)}})
        Nv = T.N.values
        rules = {
            'threshold_tau0.07_label_free': rule_summary(T, 'thr', Nv, 'Jev threshold rule tau = 0.07 (no labels)'),
            'knee_asreview': rule_summary(T, 'asr_knee', Nv, 'knee (150-record minimum) on ASReview default order, seeds 0-9'),
            'knee_hybrid': rule_summary(T, 'hybrid_knee', Nv, 'knee on hybrid order'),
            'knee_jev_only': rule_summary(T, 'jev_knee', Nv, 'knee on Jev-only order, seeds 0-9'),
            'stat_asreview': rule_summary(T, 'asr_stat', Nv, 'statistical criterion (0.95 recall, 0.95 confidence) on ASReview default order, seeds 0-9'),
            'stat_hybrid': rule_summary(T, 'hybrid_stat', Nv, 'statistical criterion on hybrid order'),
            'stat_jev_only': rule_summary(T, 'jev_stat', Nv, 'statistical criterion on Jev-only order (combined workflow: label-free ranking, labelled stop), seeds 0-9'),
            'threshold_then_stat_jev_only': rule_summary(T, 'jev_comb', Nv, 'Jev-only order read at least to tau = 0.07, then until the statistical criterion is met (addition), seeds 0-9'),
        }
        cres['rules'] = rules
        cres['stat_triggered_fraction_of_orders'] = {m: float(T[f'{m}_stat_triggered_frac'].mean()) for m in ['asr', 'hybrid', 'jev']}
        cres['stat_p_at_k95_mean'] = {m: float(T[f'{m}_stat_p_at_k95'].mean()) for m in ['asr', 'hybrid', 'jev']}
        cres['comparisons'] = {
            'stat_jev_only_minus_threshold_work': paired(T.jev_stat_work, T.thr_work, 'proportion read: statistical stop on Jev order minus threshold rule'),
            'stat_jev_only_minus_threshold_recall': paired(T.jev_stat_rec, T.thr_rec, 'recall at stop: statistical stop on Jev order minus threshold rule'),
            'stat_jev_only_minus_stat_asreview_work': paired(T.jev_stat_work, T.asr_stat_work, 'proportion read: statistical stop on Jev order minus on ASReview order'),
            'stat_hybrid_minus_stat_asreview_work': paired(T.hybrid_stat_work, T.asr_stat_work, 'proportion read: statistical stop on hybrid order minus on ASReview order'),
            'stat_asreview_minus_knee_asreview_work': paired(T.asr_stat_work, T.asr_knee_work, 'proportion read: statistical stop minus knee, ASReview order'),
            'stat_asreview_minus_knee_asreview_recall': paired(T.asr_stat_rec, T.asr_knee_rec, 'recall at stop: statistical stop minus knee, ASReview order'),
            'stat_hybrid_minus_knee_hybrid_work': paired(T.hybrid_stat_work, T.hybrid_knee_work, 'proportion read: statistical stop minus knee, hybrid order'),
            'threshold_minus_stat_asreview_work': paired(T.thr_work, T.asr_stat_work, 'proportion read: threshold rule minus statistical stop on ASReview order'),
            'comb_minus_threshold_work': paired(T.jev_comb_work, T.thr_work, 'proportion read: threshold-then-statistical minus threshold rule'),
            'comb_minus_stat_jev_only_work': paired(T.jev_comb_work, T.jev_stat_work, 'proportion read: threshold-then-statistical minus statistical stop alone, Jev order'),
            'atd_hybrid_minus_asr': paired(T.hybrid_atd, T.asr_atd, 'ATD: hybrid minus ASReview'),
            'atd_jev_minus_asr': paired(T.jev_atd, T.asr_atd, 'ATD: Jev-only minus ASReview'),
            'atd_hybrid_minus_jev': paired(T.hybrid_atd, T.jev_atd, 'ATD: hybrid minus Jev-only'),
        }
        cres['reliability_differences'] = {}
        for a, b_ in [('jev_stat', 'thr'), ('asr_stat', 'asr_knee'), ('jev_stat', 'asr_stat'), ('thr', 'asr_stat'), ('jev_comb', 'thr')]:
            ra = (T[f'{a}_rec'] >= 0.95).astype(float).values; rb = (T[f'{b_}_rec'] >= 0.95).astype(float).values
            cres['reliability_differences'][f'{a}_minus_{b_}'] = paired(ra, rb, f'reliability difference {a} minus {b_}')
        cres['atd'] = {m: {'macro': float(T[f'{m}_atd'].mean()), 'boot95': boot_mean(T[f'{m}_atd']), 'median': float(T[f'{m}_atd'].median()), 'range': [float(T[f'{m}_atd'].min()), float(T[f'{m}_atd'].max())]} for m in ['asr', 'hybrid', 'jev']}
        # macro recall curves and tau point
        cm = {m: np.mean([[r[f'x{g:.3f}'] for g in GRID] for r in curves_review if r['collection'] == coll and r['method'] == m], axis=0) for m in ['asr', 'hybrid', 'jev']}
        curves_macro[coll] = cm
        cres['tau_point_macro'] = {'mean_proportion_read': float(T.thr_work.mean()), 'mean_recall': float(T.thr_rec.mean())}
        cres['recall_at_proportion_read'] = {f'{x:.2f}': {m: float(cm[m][int(round(x / 0.005))]) for m in cm} for x in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
        # reviews under 150 records: statistical criterion versus knee
        small = T[T.N < 150]
        cres['reviews_under_150'] = {'n': int(len(small)), 'reviews': small.review.tolist(), 'knee_asr_work': small.asr_knee_work.tolist(), 'stat_asr_work': small.asr_stat_work.tolist(), 'stat_asr_rec': small.asr_stat_rec.tolist(), 'thr_work': small.thr_work.tolist(), 'thr_rec': small.thr_rec.tolist()}
        res['collections'][coll] = cres
        print(coll, 'rules:', {k: (v['reliable_reviews'], round(v['macro_work'], 4), round(v['mean_recall'], 4)) for k, v in rules.items()}, flush=True)
    pd.DataFrame(repro_rows).to_csv(os.path.join(HERE, 'reproduction_check_diffs.csv'), index=False)
    pd.DataFrame(per_order_rows).to_csv(os.path.join(HERE, 'per_order_stopping.csv'), index=False)
    pd.concat(per_review_rows.values()).to_csv(os.path.join(HERE, 'per_review_stopping.csv'), index=False)
    pd.DataFrame(curves_review).to_csv(os.path.join(HERE, 'recall_curves_per_review.csv'), index=False)
    pd.DataFrame([{'collection': coll, 'proportion_read': float(g), **{m: float(cm[m][i]) for m in cm}} for coll, cm in curves_macro.items() for i, g in enumerate(GRID)]).to_csv(os.path.join(HERE, 'recall_curves_macro.csv'), index=False)
    json.dump(p_curves, open(os.path.join(HERE, 'stat_p_curves_hybrid.json'), 'w'))
    # ---------- leave-one-review-out influence
    loo = {}
    Th = per_review_rows['heldout']; Tc = per_review_rows['clef']
    C1 = pd.read_csv(os.path.join(AN, 'AN-0001-13/c1_per_review_auc.csv')); C2 = pd.read_csv(os.path.join(AN, 'AN-0001-13/c2_per_review_auc.csv'))
    A02h = pd.read_csv(os.path.join(AN, 'AN-0001-02/per_review.csv')); A02h = A02h[A02h.collection == 'heldout_final'].set_index('review').loc[Th.review]
    A02c = pd.read_csv(os.path.join(AN, 'AN-0001-02/per_review.csv')); A02c = A02c[A02c.collection == 'clef_final'].set_index('review').loc[Tc.review]
    Z = pd.read_csv(os.path.join(ROOT, 'synergy/clef_zs_label.csv')).set_index('topic').loc[Tc.review]
    def wp(a, b):
        try: return float(wilcoxon(a, b).pvalue)
        except Exception: return float('nan')
    def f_c1(m):
        d = (C1.jev - C1.bge).values[m]; return {'C1_jev_minus_bge_auc': float(d.mean()), 'C1_wilcoxon_p': wp(C1.jev.values[m], C1.bge.values[m]), 'C1_pass': float(d.mean() >= 0.03 and wp(C1.jev.values[m], C1.bge.values[m]) < 0.01)}
    loo['C1'] = loo_ranges(C1.review.tolist(), f_c1)
    def f_c2(m):
        d1 = (C2.jev - C2.gpt4omini_lp).values[m]; d2 = (C2.claude_opus - C2.jev).values[m]; d3 = (C2.jev - C2.deepseek).values[m]
        lo1 = boot_mean(d1)[0]
        return {'C2_jev_minus_gpt4omini': float(d1.mean()), 'C2_jev_minus_gpt4omini_boot_lower': lo1, 'C2_claude_minus_jev': float(d2.mean()), 'C2_jev_minus_deepseek': float(d3.mean()),
                'C2_pass': float(lo1 > -0.02 and d2.mean() <= 0.03)}
    loo['C2'] = loo_ranges(C2.review.tolist(), f_c2)
    def f_c3(m):   # stored (v0) simulation values, as in final_eval.py
        d = (Th.hybrid_wss95_stored - Th.asr_wss95_stored).values[m]; d100 = (Th.hybrid_wss100_stored - Th.asr_wss100_stored).values[m]; p_ = wp(Th.hybrid_wss95_stored.values[m], Th.asr_wss95_stored.values[m])
        dj = (Th.hybrid_wss95_stored - Th.jev_wss95).values[m]; dja = (Th.jev_wss95 - Th.asr_wss95_stored).values[m]
        return {'C3_hybrid_minus_asr_wss95': float(d.mean()), 'C3_hybrid_minus_asr_wss100': float(d100.mean()), 'C3_wilcoxon_p': p_, 'C3_pass': float(d.mean() >= 0.05 and p_ < 0.05 and d100.mean() > 0),
                'hybrid_minus_jev_only_wss95': float(dj.mean()), 'hybrid_minus_jev_only_wilcoxon_p': wp(Th.hybrid_wss95_stored.values[m], Th.jev_wss95.values[m]), 'jev_only_minus_asr_wss95': float(dja.mean())}
    loo['C3_and_E1.04'] = loo_ranges(Th.review.tolist(), f_c3)
    def f_c4(m):
        rel = float((Th.thr_rec.values[m] >= 0.95).mean()); krel = float((Th.asr_knee_rec_stored.values[m] >= 0.95).mean()); w = float(Th.thr_work.values[m].mean()); kw = float(Th.asr_knee_work_stored.values[m].mean())
        return {'C4_threshold_reliability': rel, 'C4_threshold_mean_work': w, 'C4_knee_reliability': krel, 'C4_knee_mean_work': kw, 'C4_pass': float(rel >= 0.90 and rel >= krel and w < kw),
                'stat_asr_reliability': float((Th.asr_stat_rec.values[m] >= 0.95).mean()), 'stat_asr_mean_work': float(Th.asr_stat_work.values[m].mean()),
                'stat_jev_reliability': float((Th.jev_stat_rec.values[m] >= 0.95).mean()), 'stat_jev_mean_work': float(Th.jev_stat_work.values[m].mean()),
                'stat_hybrid_reliability': float((Th.hybrid_stat_rec.values[m] >= 0.95).mean()), 'stat_hybrid_mean_work': float(Th.hybrid_stat_work.values[m].mean()),
                'comb_jev_reliability': float((Th.jev_comb_rec.values[m] >= 0.95).mean()), 'comb_jev_mean_work': float(Th.jev_comb_work.values[m].mean()),
                'threshold_min_recall': float(Th.thr_rec.values[m].min())}
    loo['C4_and_stopping_heldout'] = loo_ranges(Th.review.tolist(), f_c4)
    def f_clef(m):
        d = (Tc.hybrid_wss95_stored - Tc.asr_wss95_stored).values[m]; d100 = (Tc.hybrid_wss100_stored - Tc.asr_wss100_stored).values[m]; auc = (Z.jev_AUC - Z.bge_AUC).values[m]
        return {'CLEF_hybrid_minus_asr_wss95': float(d.mean()), 'CLEF_hybrid_minus_asr_wss100': float(d100.mean()), 'CLEF_hybrid_minus_asr_wilcoxon_p': wp(Tc.hybrid_wss95_stored.values[m], Tc.asr_wss95_stored.values[m]),
                'CLEF_jev_minus_bge_auc': float(auc.mean()), 'CLEF_hybrid_minus_jev_only_wss95': float((Tc.hybrid_wss95_stored - Tc.jev_wss95).values[m].mean()),
                'CLEF_threshold_reliability': float((Tc.thr_rec.values[m] >= 0.95).mean()), 'CLEF_threshold_mean_work': float(Tc.thr_work.values[m].mean()), 'CLEF_threshold_min_recall': float(Tc.thr_rec.values[m].min()),
                'CLEF_knee_reliability': float((Tc.asr_knee_rec_stored.values[m] >= 0.95).mean()), 'CLEF_knee_mean_work': float(Tc.asr_knee_work_stored.values[m].mean()),
                'CLEF_stat_asr_reliability': float((Tc.asr_stat_rec.values[m] >= 0.95).mean()), 'CLEF_stat_asr_mean_work': float(Tc.asr_stat_work.values[m].mean()),
                'CLEF_stat_jev_reliability': float((Tc.jev_stat_rec.values[m] >= 0.95).mean()), 'CLEF_stat_jev_mean_work': float(Tc.jev_stat_work.values[m].mean()),
                'CLEF_stat_hybrid_reliability': float((Tc.hybrid_stat_rec.values[m] >= 0.95).mean()), 'CLEF_stat_hybrid_mean_work': float(Tc.hybrid_stat_work.values[m].mean()),
                'CLEF_comb_jev_reliability': float((Tc.jev_comb_rec.values[m] >= 0.95).mean()), 'CLEF_comb_jev_mean_work': float(Tc.jev_comb_work.values[m].mean())}
    loo['CLEF'] = loo_ranges(Tc.review.tolist(), f_clef)
    res['leave_one_review_out'] = loo
    # flat CSV of the LOO ranges
    lrows = []
    for grp, d in loo.items():
        for k, v in d.items():
            if 'loo_min' in v: lrows.append({'group': grp, 'quantity': k, 'full': v['full'], 'loo_min': v['loo_min'], 'review_min': v['review_min'], 'loo_max': v['loo_max'], 'review_max': v['review_max']})
    pd.DataFrame(lrows).to_csv(os.path.join(HERE, 'loo_influence.csv'), index=False)
    res['elapsed_seconds'] = time.time() - t_start; res['finished'] = now()
    json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
    print('finished', res['finished'], f'{res["elapsed_seconds"]:.0f}s', flush=True)

if __name__ == '__main__':
    main()
