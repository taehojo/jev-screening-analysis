# AN-0001-11 (npj round 1): threshold workflow with a within-review recall check from a simple random sample of the
# records below tau. Retrospective simulation of a screening procedure on real labels (stored scores and labels).
# Procedure fixed in REVISION_ANALYSIS_LOG.md (2026-09-28 19:48:38 EDT, wording corrected 19:50:36 EDT) before any run:
#  (1) read every record with Jev score >= tau (0.07); K = positives found there;
#  (2) draw records below tau one at a time in random order without replacement (seeds 0-9); after each draw, with k
#      positives among the n drawn, m0 = smallest number of positives below tau (population M, before the draws) at which
#      total recall would be below 0.95 = floor((K + 20 k) / 19) + 1 (exact integer form of floor((K+k)/0.95 - K) + 1);
#      P = P(X <= k), X ~ Hypergeometric(population M, m0 positives, n draws) (P = 0 if m0 > M: null impossible);
#      stop when P < 0.05, or when every record below tau has been read.
# Post hoc; no model call. Sequential testing after every draw can inflate the error rate.
import json, datetime, time, csv
import numpy as np
from scipy.stats import hypergeom
ROOT = '/N/project/AiLab/jev'; L8 = ROOT + '/review_pipeline/rounds/round_0001/revision/analysis/AN-0001-08'
TAU = 0.07; ALPHA = 0.05; SEEDS = range(10); BOOT_SEED = 20260928; B = 5000; T0 = time.time()
def wilson(k, n, z=1.959963984540054):
    ph = k / n; d = 1 + z * z / n; c = ph + z * z / (2 * n); h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)); return [float((c - h) / d), float((c + h) / d)]
def load(recf, jevf, label, or0=False):
    R = json.load(open(recf)); J = {d['id']: d['p'] for d in json.load(open(jevf)) if d.get('ok') and d.get('p') is not None}
    by = {}
    for r in R: by.setdefault(r['review'], []).append(r)
    out = {}
    for k, rs in by.items():
        if not all(r['id'] in J for r in rs): continue
        vals = [r.get(label) for r in rs]
        if any(v is None or v == '' for v in vals) and not or0: continue
        if label == 'label_ta' and any(v is None or v == '' for v in vals): continue
        y = np.array([int(v or 0) for v in vals]); p = np.array([float(J[r['id']]) for r in rs])
        if y.sum() == 0: continue
        out[k] = (p, y)
    return out
def run_review(p, y, seed):
    N = len(y); n1 = int(y.sum()); above = p >= TAU; K = int(y[above].sum()); A = int(above.sum())
    below = np.where(~above)[0]; M = len(below)
    if M == 0: return {'n_drawn': 0, 'k': 0, 'stopped_early': False, 'read': A, 'found': K, 'N': N, 'n1': n1, 'M': 0}
    rng = np.random.default_rng(seed); order = rng.permutation(below)
    yk = np.cumsum(y[order]); n = np.arange(1, M + 1)
    m0 = (K + 20 * yk) // 19 + 1
    P = np.where(m0 > M, 0.0, hypergeom.cdf(yk, M, np.minimum(m0, M), n))
    hit = np.where(P < ALPHA)[0]
    if len(hit): ns = int(hit[0]) + 1; stopped = ns < M
    else: ns = M; stopped = False
    kk = int(yk[ns - 1])
    return {'n_drawn': ns, 'k': kk, 'stopped_early': stopped, 'read': A + ns, 'found': K + kk, 'N': N, 'n1': n1, 'M': M}
colls = {'development_final': (ROOT + '/synergy/dev2000_records.json', ROOT + '/synergy/jev_dev2000.json', 'label', False),
         'heldout_final': (ROOT + '/synergy/test_records.json', ROOT + '/synergy/jev_test.json', 'label', False),
         'heldout12_ta': (ROOT + '/synergy/test_records.json', ROOT + '/synergy/jev_test.json', 'label_ta', False),
         'clef28_content': (ROOT + '/clef/clef_records.json', ROOT + '/synergy/jev_clef.json', 'label', True),
         'clef31_ta': (ROOT + '/clef/clef_records.json', ROOT + '/synergy/jev_clef.json', 'label_ta', True)}
import pandas as pd
ST = pd.read_csv(L8 + '/per_review_stopping.csv')
stored = {'heldout_final': ST[ST.collection == 'heldout'].set_index('review'), 'clef28_content': ST[ST.collection == 'clef'].set_index('review')}
out = {'analysis_id': 'AN-0001-11', 'tau': TAU, 'alpha': ALPHA, 'recall_target': 0.95, 'seeds': list(SEEDS), 'bootstrap': {'B': B, 'seed': BOOT_SEED}}
allrows = []
for cname, (rf, jf, lab, or0) in colls.items():
    D = load(rf, jf, lab, or0); rows = []
    for k in sorted(D):
        p, y = D[k]; res = [run_review(p, y, s) for s in SEEDS]
        N = len(y); n1 = int(y.sum()); A = int((p >= TAU).sum()); K = int(y[p >= TAU].sum())
        rec = np.array([r['found'] / n1 for r in res]); work = np.array([r['read'] / N for r in res])
        row = {'collection': cname, 'review': k, 'N': N, 'n1': n1, 'M_below_tau': N - A, 'pos_below_tau': n1 - K, 'thr_work': A / N, 'thr_rec': K / n1,
               'check_work_mean': float(work.mean()), 'check_rec_mean': float(rec.mean()), 'check_rec_min_over_seeds': float(rec.min()),
               'check_rel_perseed': float(np.mean(rec >= 0.95)), 'check_stopped_early_frac': float(np.mean([r['stopped_early'] for r in res])),
               'check_drawn_mean': float(np.mean([r['n_drawn'] for r in res])), 'check_drawn_frac_of_below': float(np.mean([r['n_drawn'] / max(1, r['M']) for r in res])) if N - A > 0 else None,
               'check_read_mean_records': float(np.mean([r['read'] for r in res]))}
        if cname in stored and k in stored[cname].index:
            row['comb_work_stored'] = float(stored[cname].loc[k, 'jev_stat_work']); row['comb_rec_stored'] = float(stored[cname].loc[k, 'jev_stat_rec'])
        rows.append(row)
    allrows += rows
    T = pd.DataFrame(rows); n = len(T)
    idx = np.random.default_rng(BOOT_SEED).integers(0, n, (B, n))
    def bt(v):
        v = np.asarray(v, float); m = v[idx].mean(1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
    def rule(work, recc):
        kk = int((T[recc] >= 0.95).sum())
        return {'n_reviews': n, 'n_reliable': kk, 'reliability': kk / n, 'wilson95': wilson(kk, n), 'mean_recall': float(T[recc].mean()), 'min_recall': float(T[recc].min()),
                'macro_work': float(T[work].mean()), 'macro_work_boot95': bt(T[work]), 'pooled_work': float((T[work] * T.N).sum() / T.N.sum()),
                'failing_reviews': T[T[recc] < 0.95].review.tolist()}
    s = {'threshold_alone': rule('thr_work', 'thr_rec'), 'threshold_plus_sample_check': rule('check_work_mean', 'check_rec_mean')}
    s['threshold_plus_sample_check']['reliability_per_seed_mean'] = float(T.check_rel_perseed.mean())
    s['threshold_plus_sample_check']['min_recall_any_seed'] = float(T.check_rec_min_over_seeds.min())
    s['threshold_plus_sample_check']['reviews_with_any_seed_below_0.95'] = T[T.check_rec_min_over_seeds < 0.95].review.tolist()
    s['threshold_plus_sample_check']['proportion_of_review_seed_runs_stopping_before_all_below_tau_read'] = float(T.check_stopped_early_frac.mean())
    s['threshold_plus_sample_check']['reviews_never_stopping_early'] = int((T.check_stopped_early_frac == 0).sum())
    s['threshold_plus_sample_check']['reviews_with_no_record_below_tau'] = int((T.M_below_tau == 0).sum())
    d = T.check_work_mean - T.thr_work; s['check_minus_threshold_work'] = {'mean': float(d.mean()), 'boot95': bt(d)}
    if 'comb_work_stored' in T:
        s['combined_workflow_stored'] = rule('comb_work_stored', 'comb_rec_stored')
        d2 = T.check_work_mean - T.comb_work_stored; s['check_minus_combined_work'] = {'mean': float(d2.mean()), 'boot95': bt(d2), 'check_reads_less': int((d2 < 0).sum()), 'check_reads_more': int((d2 > 0).sum())}
    out[cname] = s
    print(cname, json.dumps({kk: ({q: v for q, v in vv.items() if q in ('n_reliable', 'n_reviews', 'reliability', 'wilson95', 'min_recall', 'macro_work', 'pooled_work', 'reliability_per_seed_mean', 'min_recall_any_seed', 'proportion_of_review_seed_runs_stopping_before_all_below_tau_read', 'failing_reviews', 'reviews_with_any_seed_below_0.95')} if isinstance(vv, dict) and 'n_reviews' in vv else vv) for kk, vv in s.items()}), flush=True)
pd.DataFrame(allrows).to_csv('per_review.csv', index=False)
out['caveat'] = ('Retrospective simulation on stored scores and labels; the test is applied after every draw, which can inflate the error rate; '
                 'reliability is judged on the recall averaged over ten random samples per review (as for the knee method), and the per-seed proportion is also given.')
out['runtime_seconds'] = round(time.time() - T0, 1); out['finished'] = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
json.dump(out, open('results.json', 'w'), indent=1, default=float)
print('runtime', out['runtime_seconds'])
