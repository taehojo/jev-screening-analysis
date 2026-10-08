# AN-0001-02 (npj round 1): sensitivity of criterion C4 to the knee method with the published minimum of 1000 screened records.
# Post hoc. Reads saved labelling orders; no simulation is re-run; no model call. C4 as reported is not changed.
#
# knee_stop_min() is synergy/al_sim.py knee_stop() with one change: the first evaluation point is a parameter `smin`
# (150 in the original; 1000 for the published rule as described by Yang et al.). With smin=150 it is the original code.
# Evaluation interval max(1, N // 1000) and the stopping condition rho >= 156 - min(relret, 150) are unchanged.
# In reviews with fewer than smin records the loop is empty and all records are read (as in the original).
#
# Tango, Newcombe, McNemar functions and validate() are copied unchanged from
# review_pipeline/rounds/round_0002/revision/analysis/AN-0002-02/an02_mcnemar_knee_strata.py (Lancet round 2).
import json, math, os, sys, datetime, time, csv
from fractions import Fraction as F
import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom, norm, beta, wilcoxon
ROOT = '/N/project/AiLab/jev'
L8 = ROOT + '/review_pipeline/rounds/round_0001/revision/analysis/AN-0001-08'
AN03 = ROOT + '/review_pipeline_npj/rounds/round_0001/revision/analysis/AN-0001-03'
SEED = 20260928; B = 5000; TAU = 0.07; Z = 1.959963984540054
T0 = time.time()

def knee_stop_min(seq, y, smin):
    ys = y[np.asarray(seq)]; cum = np.cumsum(ys); N = len(seq)
    for s_ in range(smin, N + 1, max(1, N // 1000)):
        r = cum[s_ - 1]
        if r == 0: continue
        xs = np.arange(1, s_ + 1); ys_ = cum[:s_]
        d = (r * xs - s_ * ys_) / math.hypot(r, s_)
        i = int(np.argmin(d)) + 1
        ri = cum[i - 1]
        if i >= s_: continue
        rho = (ri / i) / ((r - ri + 1) / (s_ - i))
        if rho >= 156 - min(r, 150):
            return s_
    return N

# ---------------- copied from Lancet AN-0002-02 ----------------
def mcnemar_asymptotic(b, c):
    nd = b + c
    if nd == 0:
        return None, 1.0
    z = (b - c) / math.sqrt(nd)
    return z, 2 * (1 - norm.cdf(abs(z)))
def mcnemar_exact_cond(b, c):
    return min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5))
def mcnemar_midp(b, c):
    if b == c:
        return 1 - 0.5 * binom.pmf(b, b + c, 0.5)
    return min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5)) - binom.pmf(b, b + c, 0.5)
def wilson(x, n, z=Z):
    est = x / n
    A = (2 * n * est + z ** 2) / (2 * n + 2 * z ** 2)
    Bq = (z * math.sqrt(z ** 2 + 4 * n * est * (1 - est))) / (2 * n + 2 * z ** 2)
    return A - Bq, A + Bq
def tango_ci(n11, n12, n21, n22, z=Z):
    N = n11 + n12 + n21 + n22
    est = (n12 - n21) / N
    tol = 1e-7
    def p21t(d0):
        A = 2 * N
        Bq = -n12 - n21 + (2 * N - n12 + n21) * d0
        C = -n21 * d0 * (1 - d0)
        return (math.sqrt(Bq * Bq - 4 * A * C) - Bq) / (2 * A)
    def T(d0):
        return (n12 - n21 - N * d0) / math.sqrt(N * (2 * p21t(d0) + d0 * (1 - d0)))
    L = -1.0 if est == -1 else brentq(lambda d: T(d) - z, -1 + tol, 1 - tol, xtol=tol)
    U = 1.0 if est == 1 else brentq(lambda d: T(d) + z, -1 + tol, 1 - tol, xtol=tol)
    return est, L, U
def newcombe_ci(n11, n12, n21, n22, z=Z):
    N = n11 + n12 + n21 + n22
    r1, r2 = n11 + n12, n21 + n22
    c1, c2 = n11 + n21, n12 + n22
    p1, p2 = r1 / N, c1 / N
    est = p1 - p2
    l1, u1 = wilson(r1, N, z)
    l2, u2 = wilson(c1, N, z)
    if r1 == 0 or r2 == 0 or c1 == 0 or c2 == 0:
        psi = 0.0
    else:
        nprod = r1 * r2 * c1 * c2
        A = n11 * n22 - n12 * n21
        if A > N / 2:
            psi = (A - N / 2) / math.sqrt(nprod)
        elif 0 <= A <= N / 2:
            psi = 0.0
        else:
            psi = A / math.sqrt(nprod)
    L = est - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * psi * (p1 - l1) * (u2 - p2))
    U = est + math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2 - 2 * psi * (p2 - l2) * (u1 - p1))
    return est, L, U, psi
def validate():
    cavo = (59, 6, 16, 80)
    out = []
    def chk(name, got, exp, dp):
        ok = round(got, dp) == round(exp, dp) or abs(got - exp) < 0.5 * 10 ** (-dp) + 1e-12
        out.append({"check": name, "got": got, "expected": exp, "decimals": dp, "ok": bool(ok)})
    zz, p = mcnemar_asymptotic(cavo[1], cavo[2])
    chk("asymptotic P cavo", p, 0.033006, 6)
    chk("asymptotic Z cavo", zz, -2.132, 3)
    chk("exact conditional P cavo", mcnemar_exact_cond(cavo[1], cavo[2]), 0.052479, 6)
    chk("mid-P cavo", mcnemar_midp(cavo[1], cavo[2]), 0.034690, 6)
    e, L, U = tango_ci(*cavo)
    chk("Tango estimate cavo", e, -0.0621, 4); chk("Tango lower cavo", L, -0.1240, 4); chk("Tango upper cavo", U, -0.0054, 4)
    e, L, U = tango_ci(0, 3, 0, 0)
    chk("Tango estimate (0,3,0,0)", e, 1.0, 4); chk("Tango lower (0,3,0,0)", L, -0.1230, 4); chk("Tango upper (0,3,0,0)", U, 1.0, 4)
    e, L, U = tango_ci(0, 0, 3, 0)
    chk("Tango estimate (0,0,3,0)", e, -1.0, 4); chk("Tango lower (0,0,3,0)", L, -1.0, 4); chk("Tango upper (0,0,3,0)", U, 0.1230, 4)
    e, L, U, _ = newcombe_ci(*cavo)
    chk("Newcombe estimate cavo", e, -0.0621, 4); chk("Newcombe lower cavo", L, -0.1186, 4); chk("Newcombe upper cavo", U, -0.0046, 4)
    e, L, U, _ = newcombe_ci(1, 0, 0, 1)
    chk("Newcombe lower diag(1,1)", L, -0.5734, 4); chk("Newcombe upper diag(1,1)", U, 0.5734, 4)
    return out
# ----------------------------------------------------------------
def cp(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)); hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)); return [lo, hi]
out = {'analysis_id': 'AN-0001-02', 'seed': SEED, 'B': B, 'validate_paired_methods': validate()}
if not all(r['ok'] for r in out['validate_paired_methods']): raise SystemExit('STOP: paired-method validation failed')

def load_coll(recf, jevf, simf, ordf, label_or0):
    R = json.load(open(recf)); J = {d['id']: d['p'] for d in json.load(open(jevf)) if d.get('ok') and d.get('p') is not None}
    by = {}
    for r in R: by.setdefault(r['review'], []).append(r)
    sim = {}
    for d in json.load(open(simf)):
        if d.get('wss95') is not None: sim[(d['review'], d['method'], d['seed'])] = d
    orders = json.load(open(ordf))
    return by, J, sim, orders
colls = {'heldout': (ROOT + '/synergy/test_records.json', ROOT + '/synergy/jev_test.json', ROOT + '/synergy/asr_test.json', L8 + '/an08_asr_heldout_orders.json', False),
         'clef': (ROOT + '/clef/clef_records.json', ROOT + '/synergy/jev_clef.json', ROOT + '/synergy/asr_clef.json', L8 + '/an08_asr_clef_orders.json', True)}
hyb10 = {'heldout': AN03 + '/an03_hybrid_heldout_orders.json', 'clef': AN03 + '/an03_hybrid_clef_orders.json'}
per_rows = []; rep_rows = []
for coll, (recf, jevf, simf, ordf, or0) in colls.items():
    by, J, sim, orders = load_coll(recf, jevf, simf, ordf, or0)
    h10 = None
    if os.path.exists(hyb10[coll]) and os.path.exists(hyb10[coll].replace('_orders.json', '.json')):
        try:
            h10 = json.load(open(hyb10[coll]))
        except Exception as e:
            h10 = None
    revs = sorted({k.split('|')[0] for k in orders})
    res = {'n_reviews': len(revs), 'hybrid_10_seed_orders_used': h10 is not None and all(f'{k}|asr_jevblend_3|{s}' in h10 for k in revs for s in range(10))}
    for k in revs:
        rs = by[k]; y = np.array([int((r['label'] or 0) if or0 else r['label']) for r in rs]); N = len(y); n1 = int(y.sum())
        p = np.array([J[r['id']] for r in rs], float); s = p >= TAU
        row = {'collection': coll, 'review': k, 'N': N, 'n1': n1, 'thr_k': int(s.sum()), 'thr_found': int(y[s].sum())}
        for tag, meth, seeds, src in [('asr', 'asr_prior', range(10), orders), ('hybrid0', 'asr_jevblend_3', [0], orders)] + ([('hybrid10', 'asr_jevblend_3', range(10), h10)] if res['hybrid_10_seed_orders_used'] else []):
            k150, r150, k1000, r1000 = [], [], [], []
            for sd in seeds:
                seq = src[f'{k}|{meth}|{sd}']; assert len(seq) == N and len(set(seq)) == N
                a = knee_stop_min(seq, y, 150); b_ = knee_stop_min(seq, y, 1000)
                ra = int(y[np.asarray(seq[:a])].sum()); rb = int(y[np.asarray(seq[:b_])].sum())
                k150.append(a); r150.append(ra); k1000.append(b_); r1000.append(rb)
                st = sim.get((k, meth, sd))
                if st is not None and tag != 'hybrid10':
                    rep_rows.append({'collection': coll, 'review': k, 'method': meth, 'seed': sd, 'stored_knee_work': st['knee_work'], 'recomputed_knee_work': a / N, 'stored_knee_rec': st['knee_rec'], 'recomputed_knee_rec': ra / n1,
                                     'match': abs(st['knee_work'] - a / N) <= 1e-9 and abs(st['knee_rec'] - ra / n1) <= 1e-9})
            row[f'{tag}_knee150_work'] = sum(F(x, N) for x in k150) / len(k150); row[f'{tag}_knee150_rec'] = sum(F(x, n1) for x in r150) / len(r150)
            row[f'{tag}_knee1000_work'] = sum(F(x, N) for x in k1000) / len(k1000); row[f'{tag}_knee1000_rec'] = sum(F(x, n1) for x in r1000) / len(r1000)
            row[f'{tag}_knee1000_rel_perseed'] = sum(1 for x in r1000 if F(x, n1) >= F(19, 20)) / len(r1000)
            row[f'{tag}_knee1000_stopped_before_end'] = sum(1 for x in k1000 if x < N) / len(k1000)
        row['thr_work'] = F(row['thr_k'], N); row['thr_rec'] = F(row['thr_found'], n1)
        per_rows.append(row)
    out[coll] = res
# reproduction check
nrep = len(rep_rows); nmatch = sum(1 for r in rep_rows if r['match'])
out['reproduction_knee150_vs_stored'] = {'n_orders_checked': nrep, 'n_match_1e-9': nmatch, 'mismatches': [r for r in rep_rows if not r['match']]}
with open('reproduction_check.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rep_rows[0])); w.writeheader(); w.writerows(rep_rows)
# summaries
rng = np.random.default_rng(SEED)
def rel(x): return int(x >= F(19, 20))
def summarise_rule(rows, work, rec):
    n = len(rows); k = sum(rel(r[rec]) for r in rows); wl = wilson(k, n)
    wv = np.array([float(r[work]) for r in rows]); idx = np.random.default_rng(SEED).integers(0, n, (B, n)); m = wv[idx].mean(1)
    return {'n_reviews': n, 'n_reliable': k, 'reliability': k / n, 'wilson95': list(wl), 'clopper_pearson95': cp(k, n),
            'macro_work': float(wv.mean()), 'macro_work_boot95': [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
            'pooled_work': float(sum(float(r[work]) * r['N'] for r in rows) / sum(r['N'] for r in rows)),
            'mean_recall': float(np.mean([float(r[rec]) for r in rows])), 'min_recall': float(min(float(r[rec]) for r in rows)),
            'failing_reviews': [r['review'] for r in rows if not rel(r[rec])]}
def paired_rel(rows, a, b):
    ia = [rel(r[a]) for r in rows]; ib = [rel(r[b]) for r in rows]
    n11 = sum(1 for x, y in zip(ia, ib) if x and y); n12 = sum(1 for x, y in zip(ia, ib) if x and not y); n21 = sum(1 for x, y in zip(ia, ib) if (not x) and y); n22 = len(ia) - n11 - n12 - n21
    t = tango_ci(n11, n12, n21, n22); nc = newcombe_ci(n11, n12, n21, n22); za, pa = mcnemar_asymptotic(n12, n21)
    return {'n11_both': n11, 'n12_first_only': n12, 'n21_second_only': n21, 'n22_neither': n22, 'difference': (n12 - n21) / len(ia),
            'tango95': [t[1], t[2]], 'newcombe95': [nc[1], nc[2]],
            'mcnemar_asymptotic_p': pa if n12 + n21 > 0 else None, 'mcnemar_exact_p': mcnemar_exact_cond(n12, n21) if n12 + n21 > 0 else None, 'mcnemar_midp': mcnemar_midp(n12, n21) if n12 + n21 > 0 else None}
def paired_work(rows, a, b):
    d = [r[a] - r[b] for r in rows]; df = np.array([float(x) for x in d]); n = len(d); idx = np.random.default_rng(SEED).integers(0, n, (B, n)); m = df[idx].mean(1)
    res = {'mean_diff': float(df.mean()), 'boot95': [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], 'first_reads_less': sum(1 for x in d if x < 0), 'first_reads_more': sum(1 for x in d if x > 0), 'equal': sum(1 for x in d if x == 0)}
    if np.any(df != 0):
        w = wilcoxon(df); res['wilcoxon_p'] = float(w.pvalue)
        nz = df[df != 0]; ties = len(nz) - len(np.unique(np.abs(nz)))
        res['wilcoxon_method'] = 'exact' if (res['equal'] == 0 and ties == 0 and n <= 50) else ('exact (permutation)' if n <= 13 else 'asymptotic')
    return res
summ = {}
for coll in colls:
    rows_all = [r for r in per_rows if r['collection'] == coll]
    groups = {'all': rows_all, 'N<=2000': [r for r in rows_all if r['N'] <= 2000], 'N>2000': [r for r in rows_all if r['N'] > 2000], 'N<1000': [r for r in rows_all if r['N'] < 1000], 'N>=1000': [r for r in rows_all if r['N'] >= 1000]}
    sc = {}
    for gname, rows in groups.items():
        if not rows: continue
        g = {'threshold': summarise_rule(rows, 'thr_work', 'thr_rec'), 'knee150_asr': summarise_rule(rows, 'asr_knee150_work', 'asr_knee150_rec'), 'knee1000_asr': summarise_rule(rows, 'asr_knee1000_work', 'asr_knee1000_rec'),
             'knee150_hybrid_seed0': summarise_rule(rows, 'hybrid0_knee150_work', 'hybrid0_knee150_rec'), 'knee1000_hybrid_seed0': summarise_rule(rows, 'hybrid0_knee1000_work', 'hybrid0_knee1000_rec')}
        if 'hybrid10_knee1000_work' in rows[0]:
            g['knee1000_hybrid_10seeds'] = summarise_rule(rows, 'hybrid10_knee1000_work', 'hybrid10_knee1000_rec'); g['knee150_hybrid_10seeds'] = summarise_rule(rows, 'hybrid10_knee150_work', 'hybrid10_knee150_rec')
        g['paired_threshold_vs_knee1000_asr'] = {'reliability': paired_rel(rows, 'thr_rec', 'asr_knee1000_rec'), 'work': paired_work(rows, 'thr_work', 'asr_knee1000_work')}
        g['paired_threshold_vs_knee150_asr'] = {'reliability': paired_rel(rows, 'thr_rec', 'asr_knee150_rec'), 'work': paired_work(rows, 'thr_work', 'asr_knee150_work')}
        g['paired_knee1000_minus_knee150_asr'] = {'reliability': paired_rel(rows, 'asr_knee1000_rec', 'asr_knee150_rec'), 'work': paired_work(rows, 'asr_knee1000_work', 'asr_knee150_work')}
        g['asr_knee1000_reliability_per_seed_mean'] = float(np.mean([r['asr_knee1000_rel_perseed'] for r in rows]))
        g['asr_knee1000_orders_stopping_before_end'] = float(np.mean([r['asr_knee1000_stopped_before_end'] for r in rows]))
        g['reviews_with_N_below_1000_read_fully_by_knee1000'] = sum(1 for r in rows if r['N'] < 1000)
        sc[gname] = g
    t = sc['all']['threshold']; kn = sc['all']['knee1000_asr']
    sc['C4_as_written_with_knee1000_comparator'] = {
        'threshold_reliability_at_least_0.90_point_estimate': t['reliability'] >= 0.90,
        'threshold_reliability_equal_or_higher_than_knee1000': t['reliability'] >= kn['reliability'],
        'threshold_macro_work_lower_than_knee1000': t['macro_work'] < kn['macro_work'],
        'would_be_met': bool(t['reliability'] >= 0.90 and t['reliability'] >= kn['reliability'] and t['macro_work'] < kn['macro_work']),
        'note': 'C4 of the plan: reliability of at least 90% and less work than the knee method at equal or higher reliability. For CLEF the plan did not apply C4 (post-hoc collection); the same reading is given for description only.'}
    summ[coll] = sc
out['summaries'] = summ
with open('per_review.csv', 'w', newline='') as f:
    flds = list(per_rows[0].keys()) + [k for k in per_rows[-1].keys() if k not in per_rows[0]]
    w = csv.DictWriter(f, fieldnames=flds); w.writeheader()
    for r in per_rows: w.writerow({k: (float(v) if isinstance(v, F) else v) for k, v in r.items()})
out['runtime_seconds'] = round(time.time() - T0, 1); out['finished'] = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
json.dump(out, open('results.json', 'w'), indent=1, default=float)
for coll in colls:
    print(coll, 'hybrid10 used:', out[coll]['hybrid_10_seed_orders_used'])
    for g, v in summ[coll].items():
        if g.startswith('C4'): print('  ', g, v); continue
        print('  ', g, {r: (v[r]['n_reliable'], v[r]['n_reviews'], round(v[r]['macro_work'], 4)) for r in v if isinstance(v[r], dict) and 'n_reliable' in v[r]})
        print('     thr vs knee1000', v['paired_threshold_vs_knee1000_asr'])
print('reproduction', out['reproduction_knee150_vs_stored']['n_orders_checked'], out['reproduction_knee150_vs_stored']['n_match_1e-9'])
print('runtime', out['runtime_seconds'])
