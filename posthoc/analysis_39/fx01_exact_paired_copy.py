#!/usr/bin/env python
"""FX-0002-01 (audit finding G1.2.B.01): numbers of reviews higher / lower / tied and Wilcoxon signed-rank tests of every
paired comparison, recomputed in exact arithmetic.

Plan: REVISION_ANALYSIS_LOG.md, entries of 2026-09-27 04:02:58 EDT and 04:13 EDT (change of scope). Post hoc.
Reads (read only): stored simulation outputs, record files, score files, and the outputs of AN-0001-01, 02, 03, 04, 08, 13.
Writes only to the folder of this script. No model call. No bootstrap interval and no mean difference is recomputed for
reporting; the exact mean differences are compared with the stored ones as a check.

Method
- Every metric is rebuilt per review and per run as a fraction of integers (fractions.Fraction):
  WSS@95 = 1 - k95/N - 1/20, WSS@100 = 1 - k100/N, recall and proportion read = counts / n1 or / N,
  ATD = sum of discovery positions / (n1 * N), AUC = (2 * pairs won + pairs tied) / (2 * n1 * n0).
  Integers stored as floating-point ratios are recovered by multiplication; the script stops when the product is further
  than 1e-6 from an integer.
- Means over runs (random starts, random tie orders) are means of fractions.
- Paired differences are fractions; reviews are counted by the sign of the exact difference.
- Wilcoxon signed-rank test: scipy.stats.wilcoxon with default settings (two-sided, zero_method 'wilcox', method 'auto'),
  applied to the exact differences converted to floating point. Equal fractions give equal floating-point values, so zero
  differences and ties of absolute differences are kept exactly.
- Cross-check A (rule proposed by the audit): stored floating-point differences with |d| <= 1e-12 set to zero.
- Cross-check B: exact mean difference against the stored mean difference (tolerance 1e-9).
- Gap check: number of floating-point differences with 1e-12 < |d| < 1e-7.
"""
import os, sys, json, math, warnings, datetime, platform, subprocess
from fractions import Fraction as F
import numpy as np, pandas as pd, scipy
from scipy.stats import wilcoxon, rankdata

ROOT = '/N/project/AiLab/jev'; SYN = ROOT + '/synergy'
AN = ROOT + '/review_pipeline/rounds/round_0001/revision/analysis'
HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-12
SEEDS = list(range(10))
LOGF = open(os.path.join(HERE, 'fx01_exact_paired.log'), 'w', encoding='utf-8')
def now(): return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S %Z'], capture_output=True, text=True).stdout.strip()
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOGF.write(s + '\n'); LOGF.flush()
def J(p): return json.load(open(p))
started = now(); log('FX-0002-01 started', started)

# ------------------------------------------------------------------ helpers
def to_int(x, denom, what):
    v = float(x) * denom; k = round(v)
    if abs(v - k) > 1e-6:
        raise SystemExit(f'STOP: not an integer: {what}: {x} * {denom} = {v}')
    return int(k)

def sim_fracs(d):
    """exact metrics of one simulation record (one run of one method on one review)"""
    N = int(d['N']); n1 = int(d['n1'])
    out = {'wss95': 1 - F(int(d['k95']), N) - F(1, 20), 'wss100': 1 - F(int(d['k100']), N)}
    for c in ('rec@10', 'rec@20', 'rec@30'):
        out[c] = F(to_int(d[c], n1, c), n1)
    for c, den in (('knee_work', N), ('knee_rec', n1)):
        if c in d and d[c] is not None: out[c] = F(to_int(d[c], den, c), den)
    for c, v in out.items():
        if abs(float(v) - float(d[c])) > 1e-12: raise SystemExit(f'STOP: exact value differs from the stored value: {c} {float(v)} {d[c]}')
    return out

def mean_f(v):
    v = list(v); return sum(v, F(0)) / len(v)

def sim_table(paths):
    """{(review, method): {'exact': {metric: Fraction mean over runs}, 'float': {metric: float mean over runs as in the original code}, 'n_runs': int}}"""
    rows = []
    for p in paths:
        rows += [d for d in J(p) if d.get('wss95') is not None]
    ex = {}
    for d in rows:
        ex.setdefault((d['review'], d['method']), []).append(sim_fracs(d))
    D = pd.DataFrame(rows)
    cols = [c for c in ['wss95', 'wss100', 'rec@10', 'rec@20', 'rec@30', 'knee_work', 'knee_rec'] if c in D]
    g = D.groupby(['review', 'method'])[cols].mean()          # the floating-point seed mean of the original scripts
    out = {}
    for key, lst in ex.items():
        out[key] = {'exact': {c: mean_f(x[c] for x in lst) for c in lst[0]}, 'float': {c: float(g.loc[key, c]) for c in cols}, 'n_runs': len(lst)}
    return out

def load_by(path):
    by = {}
    for r in J(path): by.setdefault(r['review'], []).append(r)
    return by
def load_p(path, need_p=True):
    return {d['id']: d['p'] for d in J(path) if d.get('ok') and (d.get('p') is not None or not need_p)}

def order_metrics(seq, y):
    """integers of one reading order, with the expressions of synergy/al_sim.py metrics()"""
    y = np.asarray(y); N = len(y); n1 = int(y.sum())
    pos = np.where(y[np.asarray(seq)] == 1)[0] + 1
    k95 = int(pos[math.ceil(0.95 * n1) - 1]); k100 = int(pos[n1 - 1])
    ex = {'wss95': 1 - F(k95, N) - F(1, 20), 'wss100': 1 - F(k100, N)}
    fl = {'wss95': 1 - k95 / N - 0.05, 'wss100': 1 - k100 / N}
    for f in (0.1, 0.2, 0.3):
        c = int((pos <= math.floor(f * N)).sum()); ex[f'rec@{int(f*100)}'] = F(c, n1); fl[f'rec@{int(f*100)}'] = float(c / n1)
    return ex, fl, k95, k100

def jev_only(records, scores, label='label', reviews=None):
    """Jev ranking per review: ten random orders of tied scores (seeds 0-9, as AN-0001-02) and the order of the data file"""
    by = load_by(records); P = load_p(scores)
    out = {}
    for k, rs in by.items():
        if reviews is not None and k not in reviews: continue
        if not all(r['id'] in P for r in rs): continue
        if any(r.get(label) is None for r in rs): continue
        y = np.array([int(r[label]) for r in rs]); N = len(y); n1 = int(y.sum())
        if not (0 < n1 < N): continue
        p = np.array([float(P[r['id']]) for r in rs])
        exs, fls = [], []
        for s in SEEDS:
            rng = np.random.default_rng(s); seq = np.lexsort((rng.random(N), -p))
            e, f, _, _ = order_metrics(seq, y); exs.append(e); fls.append(f)
        e_st, f_st, _, _ = order_metrics(np.argsort(-p, kind='stable'), y)
        out[k] = {'exact': {c: mean_f(e[c] for e in exs) for c in exs[0]}, 'float': {c: float(np.array([f[c] for f in fls]).mean()) for c in fls[0]},
                  'exact_stable': e_st, 'float_stable': f_st, 'N': N, 'n1': n1}
    return out

def auc_exact(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float); n1 = int(y.sum()); n0 = len(y) - n1
    r2 = np.rint(2 * rankdata(s, method='average')).astype(np.int64)          # mid-ranks are multiples of one half
    u2 = int(r2[y == 1].sum()) - n1 * (n1 + 1)                                  # 2 * Mann-Whitney U
    return F(u2, 2 * n1 * n0)

def method_label(df, p_auto):
    """method that scipy 'auto' used, found by comparison with the explicit methods"""
    res = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try: res['p_asymptotic'] = float(wilcoxon(df, method='asymptotic').pvalue)
        except Exception as e: res['p_asymptotic'] = None
        try: res['p_exact'] = float(wilcoxon(df, method='exact').pvalue)
        except Exception as e: res['p_exact'] = None
    nz = df[df != 0]; ties = len(nz) - len(np.unique(np.abs(nz))); zeros = int((df == 0).sum())
    if zeros == 0 and ties == 0 and len(df) <= 50: lab = 'exact'
    elif len(df) <= 13: lab = 'exact (permutation; zero or tied differences)'
    else: lab = 'asymptotic'
    res['method_rule'] = lab
    res['p_auto_equals_asymptotic'] = res['p_asymptotic'] is not None and abs(res['p_asymptotic'] - p_auto) <= 1e-15
    res['p_auto_equals_exact'] = res['p_exact'] is not None and abs(res['p_exact'] - p_auto) <= 1e-15
    return res

def wil(d):
    d = np.asarray(d, float)
    if len(d) < 1 or not np.any(d != 0): return None, {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = float(wilcoxon(d).pvalue)
    return p, method_label(d, p)

ALL = []; GAP = []
def paired(analysis, key, a_ex, b_ex, a_fl, b_fl, stored=None, test=True, reviews=None, note=''):
    """a_ex, b_ex: lists of Fractions; a_fl, b_fl: floating-point values of the original computation (same order)"""
    n = len(a_ex); assert n == len(b_ex) == len(a_fl) == len(b_fl), (analysis, key)
    d = [x - y for x, y in zip(a_ex, b_ex)]
    dfl = np.asarray(a_fl, float) - np.asarray(b_fl, float)
    dex = np.array([float(v) for v in d])
    if np.max(np.abs(dex - dfl)) > 1e-9: raise SystemExit(f'STOP: exact and floating-point differences disagree: {analysis} {key} {np.max(np.abs(dex - dfl))}')
    hi = sum(1 for v in d if v > 0); lo = sum(1 for v in d if v < 0); ze = sum(1 for v in d if v == 0)
    absnz = [abs(v) for v in d if v != 0]; ties = len(absnz) - len(set(absnz))
    r = {'analysis': analysis, 'key': key, 'n': n, 'higher': hi, 'lower': lo, 'tied': ze, 'tied_absolute_differences': ties, 'mean_diff_exact': float(mean_f(d)) if n else None, 'note': note}
    # floating-point computation of the original scripts
    r['float_higher'] = int((dfl > 0).sum()); r['float_lower'] = int((dfl < 0).sum()); r['float_tied'] = int((dfl == 0).sum())
    # rule proposed by the audit
    dz = np.where(np.abs(dfl) <= TOL, 0.0, dfl)
    r['tol_higher'] = int((dz > 0).sum()); r['tol_lower'] = int((dz < 0).sum()); r['tol_tied'] = int((dz == 0).sum())
    art = [i for i in range(n) if d[i] == 0 and dfl[i] != 0]
    r['reviews_tied_but_counted_as_different'] = [reviews[i] for i in art] if reviews is not None else art
    r['residuals'] = [float(dfl[i]) for i in art]
    gap = [float(x) for x in dfl if TOL < abs(x) < 1e-7]
    if gap: GAP.append((analysis, key, gap))
    r['p_exact_arithmetic'] = r['p_float'] = r['p_tol'] = None
    if test:
        p, m = wil(dex); r['p_exact_arithmetic'] = p; r.update({'method': m.get('method_rule'), 'p_auto_equals_asymptotic': m.get('p_auto_equals_asymptotic'), 'p_auto_equals_exact': m.get('p_auto_equals_exact')})
        r['p_float'], mf = wil(dfl); r['method_float'] = mf.get('method_rule')
        r['p_tol'], _ = wil(dz)
    if stored:
        r['stored'] = stored
        if stored.get('mean_diff') is not None and abs(stored['mean_diff'] - r['mean_diff_exact']) > 1e-9:
            # values stored after rounding to four decimals by the original scripts are compared at that precision
            if not (stored.get('rounded4') and abs(round(r['mean_diff_exact'], 4) - stored['mean_diff']) < 1e-9):
                raise SystemExit(f"STOP: exact mean difference differs from the stored one: {analysis} {key} {r['mean_diff_exact']} {stored['mean_diff']}")
        sc = (stored.get('higher'), stored.get('lower'), stored.get('tied'))
        r['float_counts_equal_stored'] = all(s is None or s == f for s, f in zip(sc, (r['float_higher'], r['float_lower'], r['float_tied'])))
        if stored.get('p') is not None and r['p_float'] is not None:
            r['float_p_equals_stored'] = abs(stored['p'] - r['p_float']) <= 1e-12 * max(1.0, abs(stored['p']))
        r['counts_changed'] = any(s is not None and s != e for s, e in zip(sc, (hi, lo, ze)))
        r['p_changed'] = (stored.get('p') is not None and r['p_exact_arithmetic'] is not None and abs(stored['p'] - r['p_exact_arithmetic']) > 1e-12 * max(1.0, abs(stored['p'])))
    r['per_review_diff_exact'] = [str(v) for v in d]; r['per_review_diff_float'] = [float(v) for v in d]; r['reviews'] = list(reviews) if reviews is not None else None   # AN-0001-08 (npj): per-review exact differences stored
    ALL.append(r); return r

def fmt_p(p):
    if p is None: return None
    return '<0.0001' if p < 0.0001 else f'{p:.2g}'

OUT = {'analysis_id': 'FX-0002-01', 'started': started, 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__, 'executable': sys.executable},
       'settings': {'tolerance_of_cross_check_A': TOL, 'wilcoxon': 'scipy.stats.wilcoxon defaults (two-sided, zero_method wilcox, correction False, method auto) on exact differences converted to floating point',
                    'tie_seeds_jev_ranking': SEEDS}}

# ------------------------------------------------------------------ inputs shared by the blocks
SIM = {'heldout': sim_table([SYN + '/asr_test.json', AN + '/AN-0001-03/an03_asr_heldout.json']),
       'clef': sim_table([SYN + '/asr_clef.json', AN + '/AN-0001-03/an03_asr_clef.json']),
       'development': sim_table([SYN + '/asr_dev.json', AN + '/AN-0001-03/an03_asr_dev.json']),
       'heldout_ta': sim_table([SYN + '/asr_test_ta.json']),
       'prereg': sim_table([SYN + '/al_test_prereg.json'])}
log('simulation tables', {k: len(v) for k, v in SIM.items()}, now())
ta_revs = sorted({d['review'] for d in J(SYN + '/asr_test_ta.json')}); clef_revs = sorted({d['review'] for d in J(SYN + '/asr_clef.json')})
JO = {'heldout': jev_only(SYN + '/test_records.json', SYN + '/jev_test.json'),
      'heldout_ta': jev_only(SYN + '/test_records.json', SYN + '/jev_test.json', label='label_ta', reviews=set(ta_revs)),
      'clef': jev_only(ROOT + '/clef/clef_records.json', SYN + '/jev_clef.json', reviews=set(clef_revs)),
      'development': jev_only(SYN + '/dev2000_records.json', SYN + '/jev_dev2000.json')}
log('Jev rankings', {k: len(v) for k, v in JO.items()}, now())
# check of the Jev rankings against the stored per-review values of AN-0001-02
P02 = pd.read_csv(AN + '/AN-0001-02/per_review.csv')
chk = {}
for coll, c02 in [('heldout', 'heldout_final'), ('heldout_ta', 'heldout_ta'), ('clef', 'clef_final')]:
    T = P02[P02.collection == c02].set_index('review'); mx = 0.0
    assert set(T.index) == set(JO[coll]), (coll, len(T), len(JO[coll]))
    for k in T.index:
        for c in ['wss95', 'wss100', 'rec@10', 'rec@20', 'rec@30']:
            mx = max(mx, abs(T.loc[k, f'jevonly_{c}'] - JO[coll][k]['float'][c]))
        for c in ['wss95', 'wss100']:
            mx = max(mx, abs(T.loc[k, f'jevonly_stable_{c}'] - JO[coll][k]['float_stable'][c]))
    chk[coll] = mx
    if mx > 1e-12: raise SystemExit(f'STOP: Jev ranking differs from AN-0001-02: {coll} {mx}')
OUT['check_jev_ranking_against_AN02_max_abs_diff'] = chk; log('Jev ranking against AN-0001-02', chk)

def vals(coll, method, metric, revs, which='exact'):
    if method == 'jev_only': return [JO[coll][k][which][metric] for k in revs]
    if method == 'jev_only_stable': return [JO[coll][k][which + '_stable'][metric] for k in revs]
    src = SIM[coll]
    return [src[(k, method)][which][metric] for k in revs]

# ------------------------------------------------------------------ 1. AN-0001-02 (table S14)
R2 = J(AN + '/AN-0001-02/results.json')['collections']
TAG = {'hybrid': 'asr_jevblend_3', 'asr': 'asr_prior', 'asr_jevstart': 'asr_jev', 'pseudo': 'asr_pseudo_10_50', 'jevonly': 'jev_only'}
n02 = 0
for c02, coll in [('heldout_final', 'heldout'), ('heldout_ta', 'heldout_ta'), ('clef_final', 'clef')]:
    revs = R2[c02]['reviews']
    for ckey, comp in R2[c02]['comparisons'].items():
        if ckey == 'hybrid_minus_jevonly_stable': a, b = 'asr_jevblend_3', 'jev_only_stable'
        else:
            a, b = ckey.split('_minus_'); a, b = TAG[a], TAG[b]
        for metric, e in comp.items():
            st = {'mean_diff': e['diff'], 'higher': e['wins_a_gt_b'], 'lower': e['losses_a_lt_b'], 'tied': e['ties'], 'p': (e.get('wilcoxon') or {}).get('p_auto'), 'method': (e.get('wilcoxon') or {}).get('method_auto')}
            paired('AN-0001-02', f'{c02}|{ckey}|{metric}', vals(coll, a, metric, revs), vals(coll, b, metric, revs), vals(coll, a, metric, revs, 'float'), vals(coll, b, metric, revs, 'float'),
                   stored=st, test=e['n'] > 5, reviews=revs)
            n02 += 1
log('AN-0001-02 comparisons', n02, now())

# ------------------------------------------------------------------ 2. AN-0001-03 (table S37)
R3 = J(AN + '/AN-0001-03/results.json')['collections']; n03 = 0
for coll in ['development', 'heldout', 'clef']:
    revs = R3[coll]['reviews']
    for ckey, comp in R3[coll]['comparisons'].items():
        a, b = ckey.split('_minus_')
        for metric, e in comp.items():
            st = {'mean_diff': e['mean_diff'], 'higher': e['wins'], 'lower': e['losses'], 'tied': e['ties'], 'p': e.get('wilcoxon_p'), 'method': e.get('wilcoxon_method_auto')}
            paired('AN-0001-03', f'{coll}|{ckey}|{metric}', vals(coll, a, metric, revs), vals(coll, b, metric, revs), vals(coll, a, metric, revs, 'float'), vals(coll, b, metric, revs, 'float'),
                   stored=st, test=(e['n'] > 5), reviews=revs)
            n03 += 1
log('AN-0001-03 comparisons', n03, now())

# ------------------------------------------------------------------ 3. AN-0001-08 (table S17, section 9.7, leave-one-out) and AN-0001-04 (threshold against knee)
PO = pd.read_csv(AN + '/AN-0001-08/per_order_stopping.csv'); PR = pd.read_csv(AN + '/AN-0001-08/per_review_stopping.csv')
R8 = J(AN + '/AN-0001-08/results.json'); R4 = J(AN + '/AN-0001-04/results.json')['collections']
EX8 = {}; near95 = []
for coll in ['heldout', 'clef']:
    T = PR[PR.collection == coll].set_index('review'); O = PO[PO.collection == coll]
    ex = {}
    for k in T.index:
        N = int(T.loc[k, 'N']); n1 = int(T.loc[k, 'n1']); e = {}
        e['thr_work'] = F(int(T.loc[k, 'thr_k']), N); e['thr_rec'] = F(to_int(T.loc[k, 'thr_rec'], n1, 'thr_rec'), n1)
        for me, tag in [('asr_prior', 'asr'), ('asr_jevblend_3', 'hybrid'), ('jev_only', 'jev')]:
            S = O[(O.review == k) & (O.method == me)]
            assert len(S) == (1 if me == 'asr_jevblend_3' else 10), (coll, k, me, len(S))
            e[f'{tag}_stat_work'] = mean_f(F(int(x), N) for x in S.stat_k); e[f'{tag}_knee_work'] = mean_f(F(int(x), N) for x in S.knee_k)
            e[f'{tag}_stat_rec'] = mean_f(F(to_int(x, n1, 'stat_rec'), n1) for x in S.stat_rec); e[f'{tag}_knee_rec'] = mean_f(F(to_int(x, n1, 'knee_rec'), n1) for x in S.knee_rec)
            e[f'{tag}_atd'] = mean_f(F(to_int(x, n1 * N, 'atd'), n1 * N) for x in S.atd)
            e[f'{tag}_wss95'] = mean_f(1 - F(int(x), N) - F(1, 20) for x in S.k95)
            if me == 'jev_only':
                e['jev_comb_work'] = mean_f(F(int(x), N) for x in S.comb_k); e['jev_comb_rec'] = mean_f(F(to_int(x, n1, 'comb_rec'), n1) for x in S.comb_rec)
        for c, v in e.items():
            if c in T.columns and abs(float(v) - float(T.loc[k, c])) > 1e-12: raise SystemExit(f'STOP: AN-0001-08 value differs: {coll} {k} {c} {float(v)} {T.loc[k, c]}')
            if c.endswith('_rec') and abs(float(v) - 0.95) < 1e-9 and v != F(19, 20): near95.append((coll, k, c, float(v)))
        # stored seed means of the original simulations (used for the leave-one-out values)
        for tag, me in [('asr', 'asr_prior'), ('hybrid', 'asr_jevblend_3')]:
            for q in ['wss95', 'wss100', 'knee_work', 'knee_rec']:
                e[f'{tag}_{q}_stored'] = SIM[coll][(k, me)]['exact'][q]
                if abs(float(e[f'{tag}_{q}_stored']) - float(T.loc[k, f'{tag}_{q}_stored'])) > 1e-12: raise SystemExit(f'STOP: stored seed mean differs: {coll} {k} {tag} {q}')
        ex[k] = e
    EX8[coll] = ex
    revs = list(T.index)
    PAIRS = {'stat_jev_only_minus_threshold_work': ('jev_stat_work', 'thr_work'), 'stat_jev_only_minus_threshold_recall': ('jev_stat_rec', 'thr_rec'),
             'stat_jev_only_minus_stat_asreview_work': ('jev_stat_work', 'asr_stat_work'), 'stat_hybrid_minus_stat_asreview_work': ('hybrid_stat_work', 'asr_stat_work'),
             'stat_asreview_minus_knee_asreview_work': ('asr_stat_work', 'asr_knee_work'), 'stat_asreview_minus_knee_asreview_recall': ('asr_stat_rec', 'asr_knee_rec'),
             'stat_hybrid_minus_knee_hybrid_work': ('hybrid_stat_work', 'hybrid_knee_work'), 'threshold_minus_stat_asreview_work': ('thr_work', 'asr_stat_work'),
             'comb_minus_threshold_work': ('jev_comb_work', 'thr_work'), 'comb_minus_stat_jev_only_work': ('jev_comb_work', 'jev_stat_work'),
             'atd_hybrid_minus_asr': ('hybrid_atd', 'asr_atd'), 'atd_jev_minus_asr': ('jev_atd', 'asr_atd'), 'atd_hybrid_minus_jev': ('hybrid_atd', 'jev_atd')}
    cm = R8['collections'][coll]['comparisons']
    assert set(cm) == set(PAIRS), sorted(set(cm) ^ set(PAIRS))
    for key, (a, b) in PAIRS.items():
        e = cm[key]; st = {'mean_diff': e['mean_diff'], 'higher': e['wins'], 'lower': e['losses'], 'tied': e['ties'], 'p': e.get('wilcoxon_p')}
        paired('AN-0001-08', f'{coll}|{key}', [ex[k][a] for k in revs], [ex[k][b] for k in revs], [float(T.loc[k, a]) for k in revs], [float(T.loc[k, b]) for k in revs], stored=st, test=True, reviews=revs)
    # differences in reliability: indicators of recall >= 0.95, judged on exact seed-mean recall
    for key, e in R8['collections'][coll]['reliability_differences'].items():
        a, b = key.split('_minus_')
        ia = [F(int(ex[k][f'{a}_rec'] >= F(19, 20))) for k in revs]; ib = [F(int(ex[k][f'{b}_rec'] >= F(19, 20))) for k in revs]
        fa = [float(T.loc[k, f'{a}_rec'] >= 0.95) for k in revs]; fb = [float(T.loc[k, f'{b}_rec'] >= 0.95) for k in revs]
        if [float(x) for x in ia] != fa or [float(x) for x in ib] != fb: log('NOTE reliability indicator differs between exact and floating-point recall', coll, key)
        st = {'mean_diff': e['mean_diff'], 'higher': e['wins'], 'lower': e['losses'], 'tied': e['ties'], 'p': e.get('wilcoxon_p')}
        paired('AN-0001-08', f'{coll}|reliability|{key}', ia, ib, fa, fb, stored=st, test=True, reviews=revs)
OUT['seed_mean_recalls_within_1e-9_of_0.95_but_not_equal'] = near95
log('AN-0001-08 done; recalls near 0.95 that are not exactly 0.95:', near95, now())

# AN-0001-04: threshold minus knee (proportion read), counts only (no test was stored)
def thr_exact(records, scores):
    by = load_by(records); P = load_p(scores); out = {}
    for k, rs in by.items():
        if not all(r['id'] in P for r in rs): continue
        if any(r.get('label') is None for r in rs): continue
        y = np.array([int(r['label']) for r in rs]); p = np.array([float(P[r['id']]) for r in rs]); N = len(y); n1 = int(y.sum())
        if not (0 < n1 < N): continue
        s = p >= 0.07
        out[k] = {'thr_work': F(int(s.sum()), N), 'thr_work_float': float(s.mean()), 'thr_rec': F(int(y[s].sum()), n1)}
    return out
THR = {'heldout': thr_exact(SYN + '/test_records.json', SYN + '/jev_test.json'), 'clef': thr_exact(ROOT + '/clef/clef_records.json', SYN + '/jev_clef.json'),
       'development': thr_exact(SYN + '/dev2000_records.json', SYN + '/jev_dev2000.json')}
for coll, pr in [('heldout', 'per_review_heldout.csv'), ('clef', 'per_review_clef.csv'), ('development', 'per_review_development.csv')]:
    T4 = pd.read_csv(AN + '/AN-0001-04/' + pr); T4 = T4[T4.thr_work.notna()].set_index('review'); revs = list(T4.index)
    assert set(revs) == set(THR[coll]), (coll, len(revs), len(THR[coll]))
    for tag, me in [('asr', 'asr_prior'), ('hybrid', 'asr_jevblend_3')]:
        key = f'paired_threshold_vs_knee_{tag}'
        if key not in R4[coll]: continue
        e = R4[coll][key]
        a = [THR[coll][k]['thr_work'] for k in revs]; b = [SIM[coll][(k, me)]['exact']['knee_work'] for k in revs]
        fa = [float(T4.loc[k, 'thr_work']) for k in revs]; fb = [float(T4.loc[k, f'{tag}_knee_work']) for k in revs]
        st = {'mean_diff': e['work_diff_threshold_minus_knee_macro'], 'higher': e['reviews_threshold_reads_more'], 'lower': e['reviews_threshold_reads_less'], 'tied': None, 'p': None}
        paired('AN-0001-04', f'{coll}|{key}|work', a, b, fa, fb, stored=st, test=True, reviews=revs, note='no test was stored by AN-0001-04; the p value is new')
        # reliability indicators in exact arithmetic
        ra = [int(THR[coll][k]['thr_rec'] >= F(19, 20)) for k in revs]; rb = [int(SIM[coll][(k, me)]['exact']['knee_rec'] >= F(19, 20)) for k in revs]
        only_a = sum(1 for x, y_ in zip(ra, rb) if x == 1 and y_ == 0); only_b = sum(1 for x, y_ in zip(ra, rb) if x == 0 and y_ == 1)
        ALL.append({'analysis': 'AN-0001-04', 'key': f'{coll}|{key}|reliability', 'n': len(revs), 'higher': only_a, 'lower': only_b, 'tied': len(revs) - only_a - only_b,
                    'stored': {'higher': e['reviews_threshold_reliable_knee_not'], 'lower': e['reviews_knee_reliable_threshold_not']},
                    'counts_changed': (only_a, only_b) != (e['reviews_threshold_reliable_knee_not'], e['reviews_knee_reliable_threshold_not']), 'p_exact_arithmetic': None, 'note': 'reliable reviews: threshold only / knee only'})
log('AN-0001-04 done', now())

# ------------------------------------------------------------------ 4. AN-0001-13 (table S31, section 9.11) and the counts stored by the original analysis
W13 = pd.read_csv(AN + '/AN-0001-13/wilcoxon_table.csv').set_index('test')
def st13(name, extra=None):
    r = W13.loc[name]; s = {'mean_diff': float(r['mean_diff']), 'higher': int(r['wins_a_gt_b']), 'lower': int(r['losses_a_lt_b']), 'tied': int(r['zeros']), 'p': None if pd.isna(r['p_auto']) else float(r['p_auto']), 'method': r['method_auto_inferred'],
                            'stored_by_the_original_analysis_p': None if pd.isna(r['stored_p']) else float(r['stored_p'])}
    if extra: s.update(extra)
    return s
used13 = set()
def t13(name, a_ex, b_ex, a_fl, b_fl, revs):
    used13.add(name); return paired('AN-0001-13', name, a_ex, b_ex, a_fl, b_fl, stored=st13(name), test=True, reviews=revs)
# C1: AUC of Jev and of bge-base on the 23 held-out reviews
by = load_by(SYN + '/test_records.json'); Pj = load_p(SYN + '/jev_test.json')
C1 = pd.read_csv(AN + '/AN-0001-13/c1_per_review_auc.csv').set_index('review'); revs = list(C1.index)
ax, bx = [], []
for k in revs:
    v = by[k]; y = [int(r['label']) for r in v]; S = J(f'{SYN}/scores_zs/test_{k}.json')
    ax.append(auc_exact(y, [Pj[r['id']] for r in v])); bx.append(auc_exact(y, S['bge']))
    assert abs(float(ax[-1]) - C1.loc[k, 'jev']) < 1e-12 and abs(float(bx[-1]) - C1.loc[k, 'bge']) < 1e-12, k
t13('C1 AUC Jev vs bge-base, 23 held-out reviews', ax, bx, list(C1.jev), list(C1.bge), revs)
C1EX = dict(zip(revs, zip(ax, bx)))
# C2: 1999 common records
R = J(SYN + '/test_records.json'); ids = set(J(SYN + '/test_cmp_ids.json')); lab = {r['id']: int(r['label']) for r in R}; rev = {r['id']: r['review'] for r in R}
M = {'jev': Pj}
for nm, f in [('gpt4omini_lp', 'test_cmp_gpt4omini_lp.json'), ('deepseek', 'test_cmp_deepseek.json'), ('claude_opus', 'test_cmp_claude-opus.json')]: M[nm] = load_p(f'{SYN}/{f}')
common = [i for i in ids if all(i in m for m in M.values())]
C2 = pd.read_csv(AN + '/AN-0001-13/c2_per_review_auc.csv').set_index('review'); revs2 = list(C2.index); E2 = {}
for k in revs2:
    ii = [i for i in common if rev[i] == k]; y = [lab[i] for i in ii]
    E2[k] = {m: auc_exact(y, [M[m][i] for i in ii]) for m in M}
    for m in M: assert abs(float(E2[k][m]) - C2.loc[k, m]) < 1e-12, (k, m)
OUT['C2_n_common_records'] = len(common)
t13('C2 AUC Jev vs GPT-4o-mini (log-probability), 23 reviews, 1999 common records (descriptive, no test in v0)', [E2[k]['jev'] for k in revs2], [E2[k]['gpt4omini_lp'] for k in revs2], list(C2.jev), list(C2.gpt4omini_lp), revs2)
t13('C2 AUC Claude Opus vs Jev, 23 reviews (descriptive, no test in v0)', [E2[k]['claude_opus'] for k in revs2], [E2[k]['jev'] for k in revs2], list(C2.claude_opus), list(C2.jev), revs2)
t13('C2 AUC Jev vs DeepSeek, 23 reviews (descriptive, no test in v0)', [E2[k]['jev'] for k in revs2], [E2[k]['deepseek'] for k in revs2], list(C2.jev), list(C2.deepseek), revs2)
# C3, comparators named in the plan, P1, P4 (held-out simulations)
hrevs = sorted({k for (k, m) in SIM['heldout'] if m == 'asr_jevblend_3'})
def simpair(name, coll, a, b, metric, revs_):
    return t13(name, vals(coll, a, metric, revs_), vals(coll, b, metric, revs_), vals(coll, a, metric, revs_, 'float'), vals(coll, b, metric, revs_, 'float'), revs_)
simpair('C3 WSS@95 hybrid vs ASReview default, 23 held-out reviews', 'heldout', 'asr_jevblend_3', 'asr_prior', 'wss95', hrevs)
simpair('C3 WSS@100 hybrid vs ASReview default, 23 held-out reviews (no test in v0)', 'heldout', 'asr_jevblend_3', 'asr_prior', 'wss100', hrevs)
for m, tag in [('al|lr|oracle|0|0', 'logistic_regression'), ('al|nb|oracle|0|0', 'naive_bayes')]:
    for metric, nm in [('wss95', f'Hybrid vs prespecified comparator {tag}, WSS@95, 23 held-out reviews'), ('wss100', f'Hybrid vs prespecified comparator {tag}, WSS@100, 23 held-out reviews (no test in v0)')]:
        t13(nm, vals('heldout', 'asr_jevblend_3', metric, hrevs), [SIM['prereg'][(k, m)]['exact'][metric] for k in hrevs], vals('heldout', 'asr_jevblend_3', metric, hrevs, 'float'), [SIM['prereg'][(k, m)]['float'][metric] for k in hrevs], hrevs)
for nm, a, b, metric in [('P1 WSS@95 hybrid vs weak supervision, 23', 'asr_jevblend_3', 'asr_pseudo_10_50', 'wss95'), ('P1 WSS@95 weak supervision vs ASReview, 23', 'asr_pseudo_10_50', 'asr_prior', 'wss95'),
                         ('P1 WSS@100 hybrid vs weak supervision, 23', 'asr_jevblend_3', 'asr_pseudo_10_50', 'wss100'), ('P1 WSS@100 weak supervision vs ASReview, 23', 'asr_pseudo_10_50', 'asr_prior', 'wss100'),
                         ('P4 WSS@95 strongest preset plus Jev vs strongest preset, 23', 'h3_jevblend_3', 'h3_prior', 'wss95'), ('P4 WSS@95 strongest preset vs default, 23', 'h3_prior', 'asr_prior', 'wss95'),
                         ('P4 WSS@100 strongest preset plus Jev vs strongest preset, 23', 'h3_jevblend_3', 'h3_prior', 'wss100')]:
    simpair(nm, 'heldout', a, b, metric, hrevs)
simpair('P3 title-abstract labels WSS@95 hybrid vs ASReview, 12', 'heldout_ta', 'asr_jevblend_3', 'asr_prior', 'wss95', ta_revs)
simpair('P3 title-abstract labels WSS@100 hybrid vs ASReview, 12', 'heldout_ta', 'asr_jevblend_3', 'asr_prior', 'wss100', ta_revs)
# CLEF: AUC of Jev and of bge-base for the two label levels; simulations
byc = load_by(ROOT + '/clef/clef_records.json'); Pc = {d['id']: d['p'] for d in J(SYN + '/jev_clef.json') if d.get('ok')}
for labk, csvf, nm in [('label', 'clef_zs_label.csv', 'CLEF AUC Jev vs bge-base, final inclusion, 28 reviews'), ('label_ta', 'clef_zs_label_ta.csv', 'CLEF AUC Jev vs bge-base, title-abstract labels, 31 reviews')]:
    T = pd.read_csv(f'{SYN}/{csvf}').set_index('topic'); revs_ = list(T.index); ax, bx = [], []
    for k in revs_:
        v = byc[k]; y = [int(r[labk] or 0) for r in v]; S = J(f'{SYN}/scores_zs/clef_{k}.json')
        ax.append(auc_exact(y, [Pc[r['id']] for r in v])); bx.append(auc_exact(y, S['bge']))
        assert abs(float(ax[-1]) - T.loc[k, 'jev_AUC']) < 1e-12 and abs(float(bx[-1]) - T.loc[k, 'bge_AUC']) < 1e-12, (k, labk)
    t13(nm, ax, bx, list(T.jev_AUC), list(T.bge_AUC), revs_)
for a, b in [('asr_jevblend_3', 'asr_prior'), ('asr_jevblend_3', 'asr_pseudo_10_50')]:
    for metric in ['wss95', 'wss100']:
        simpair(f'CLEF {metric} {a} vs {b}, 28 reviews', 'clef', a, b, metric, clef_revs)
# P2: scoring modes on the 2002-record subset
Rmap = {r['id']: r for r in R}; L = lambda f: {d['id']: d['p'] for d in J(f'{SYN}/{f}') if d.get('ok')}
S1, B10s, B10f = L('jev_testcmp_single.json'), L('jev_testcmp_b10.json'), L('jev_test.json')
common2 = [i for i in ids if i in S1 and i in B10s and i in B10f]; P2 = {}
from sklearn.metrics import roc_auc_score
for k in sorted({Rmap[i]['review'] for i in common2}):
    ii = [i for i in common2 if Rmap[i]['review'] == k]; y = [int(Rmap[i]['label']) for i in ii]
    if 0 < sum(y) < len(y):
        P2[k] = {m: (auc_exact(y, [S[i] for i in ii]), float(roc_auc_score(y, [S[i] for i in ii]))) for m, S in [('single', S1), ('b10_subset', B10s), ('b10_full', B10f)]}
        for m in P2[k]: assert abs(float(P2[k][m][0]) - P2[k][m][1]) < 1e-12
r2 = list(P2)
t13('P2 AUC single-record vs full-run batches, 23', [P2[k]['single'][0] for k in r2], [P2[k]['b10_full'][0] for k in r2], [P2[k]['single'][1] for k in r2], [P2[k]['b10_full'][1] for k in r2], r2)
t13('P2 AUC subset batches vs full-run batches, 23', [P2[k]['b10_subset'][0] for k in r2], [P2[k]['b10_full'][0] for k in r2], [P2[k]['b10_subset'][1] for k in r2], [P2[k]['b10_full'][1] for k in r2], r2)
assert used13 == set(W13.index), sorted(set(W13.index) - used13)
log('AN-0001-13 tests', len(used13), now())

# counts stored by the original analysis (quoted in the manuscript and in appendix section 6)
FT = J(SYN + '/final_test.json'); FD = J(SYN + '/final_dev.json'); PT = J(SYN + '/post_test.json'); PC = J(SYN + '/prereg_comparator_test.json'); CL = J(SYN + '/clef_results.json')
byd = load_by(SYN + '/dev2000_records.json'); Pd = load_p(SYN + '/jev_dev2000.json'); Zd = pd.read_csv(SYN + '/zs_auc_dev.csv').set_index('review')
drevs = [k for k, v in byd.items() if all(r['id'] in Pd for r in v) and 0 < sum(int(r['label']) for r in v) < len(v)]
ax, bx, af, bf = [], [], [], []
for k in drevs:
    v = byd[k]; y = [int(r['label']) for r in v]; S = J(f'{SYN}/scores_zs/dev_{k}.json')
    ax.append(auc_exact(y, [Pd[r['id']] for r in v])); bx.append(auc_exact(y, S['bge'])); af.append(float(roc_auc_score(y, [Pd[r['id']] for r in v]))); bf.append(float(roc_auc_score(y, S['bge'])))
paired('original', 'final_dev.json|C1 AUC Jev minus bge-base, development reviews', ax, bx, af, bf, stored={'mean_diff': FD['C1']['diff'], 'rounded4': True, 'higher': FD['C1']['jev_wins'], 'lower': None, 'tied': None, 'p': FD['C1']['wilcoxon_p']}, reviews=drevs)
dv = sorted({k for (k, m) in SIM['development'] if m == 'asr_jevblend_3'})
paired('original', 'final_dev.json|C3 WSS@95 hybrid minus ASReview, development reviews', vals('development', 'asr_jevblend_3', 'wss95', dv), vals('development', 'asr_prior', 'wss95', dv), vals('development', 'asr_jevblend_3', 'wss95', dv, 'float'), vals('development', 'asr_prior', 'wss95', dv, 'float'),
       stored={'mean_diff': FD['C3']['diff'], 'rounded4': True, 'higher': FD['C3']['wins'], 'lower': None, 'tied': None, 'p': FD['C3']['wilcoxon_p']}, reviews=dv)
ORIG = [('final_test.json|C1', 'C1 AUC Jev vs bge-base, 23 held-out reviews', {'higher': FT['C1']['jev_wins'], 'p': FT['C1']['wilcoxon_p']}),
        ('final_test.json|C3', 'C3 WSS@95 hybrid vs ASReview default, 23 held-out reviews', {'higher': FT['C3']['wins'], 'p': FT['C3']['wilcoxon_p']}),
        ('prereg_comparator_test.json|lr', 'Hybrid vs prespecified comparator logistic_regression, WSS@95, 23 held-out reviews', {'higher': PC['al|lr|oracle|0|0']['wins'], 'p': PC['al|lr|oracle|0|0']['p']}),
        ('prereg_comparator_test.json|nb', 'Hybrid vs prespecified comparator naive_bayes, WSS@95, 23 held-out reviews', {'higher': PC['al|nb|oracle|0|0']['wins'], 'p': PC['al|nb|oracle|0|0']['p']}),
        ('clef_results.json|zeroshot_label', 'CLEF AUC Jev vs bge-base, final inclusion, 28 reviews', {'higher': CL['zeroshot_label']['jev_vs_bge_auc']['wins'], 'p': CL['zeroshot_label']['jev_vs_bge_auc']['p']}),
        ('clef_results.json|zeroshot_label_ta', 'CLEF AUC Jev vs bge-base, title-abstract labels, 31 reviews', {'higher': CL['zeroshot_label_ta']['jev_vs_bge_auc']['wins'], 'p': CL['zeroshot_label_ta']['jev_vs_bge_auc']['p']}),
        ('post_test.json|P1_wss95[0]', 'P1 WSS@95 hybrid vs weak supervision, 23', {'higher': PT['P1_wss95'][0]['wins'], 'lower': PT['P1_wss95'][0]['losses'], 'p': PT['P1_wss95'][0]['wilcoxon_p']}),
        ('post_test.json|P1_wss95[1]', 'P1 WSS@95 weak supervision vs ASReview, 23', {'higher': PT['P1_wss95'][1]['wins'], 'lower': PT['P1_wss95'][1]['losses'], 'p': PT['P1_wss95'][1]['wilcoxon_p']}),
        ('post_test.json|P1_wss100[0]', 'P1 WSS@100 hybrid vs weak supervision, 23', {'higher': PT['P1_wss100'][0]['wins'], 'lower': PT['P1_wss100'][0]['losses'], 'p': PT['P1_wss100'][0]['wilcoxon_p']}),
        ('post_test.json|P1_wss100[1]', 'P1 WSS@100 weak supervision vs ASReview, 23', {'higher': PT['P1_wss100'][1]['wins'], 'lower': PT['P1_wss100'][1]['losses'], 'p': PT['P1_wss100'][1]['wilcoxon_p']}),
        ('post_test.json|P4_wss95[0]', 'P4 WSS@95 strongest preset plus Jev vs strongest preset, 23', {'higher': PT['P4_wss95'][0]['wins'], 'lower': PT['P4_wss95'][0]['losses'], 'p': PT['P4_wss95'][0]['wilcoxon_p']}),
        ('post_test.json|P4_wss95[1]', 'P4 WSS@95 strongest preset vs default, 23', {'higher': PT['P4_wss95'][1]['wins'], 'lower': PT['P4_wss95'][1]['losses'], 'p': PT['P4_wss95'][1]['wilcoxon_p']}),
        ('post_test.json|P4_wss100[0]', 'P4 WSS@100 strongest preset plus Jev vs strongest preset, 23', {'higher': PT['P4_wss100'][0]['wins'], 'lower': PT['P4_wss100'][0]['losses'], 'p': PT['P4_wss100'][0]['wilcoxon_p']}),
        ('post_test.json|P3_ta_wss95', 'P3 title-abstract labels WSS@95 hybrid vs ASReview, 12', {'higher': PT['P3_ta_wss95']['wins'], 'lower': PT['P3_ta_wss95']['losses'], 'p': PT['P3_ta_wss95']['wilcoxon_p']}),
        ('post_test.json|P3_ta_wss100', 'P3 title-abstract labels WSS@100 hybrid vs ASReview, 12', {'higher': PT['P3_ta_wss100']['wins'], 'lower': PT['P3_ta_wss100']['losses'], 'p': PT['P3_ta_wss100']['wilcoxon_p']}),
        ('post_test.json|P2.single_minus_full', 'P2 AUC single-record vs full-run batches, 23', {'higher': PT['P2']['single_minus_full']['wins'], 'lower': PT['P2']['single_minus_full']['losses'], 'p': PT['P2']['single_minus_full']['wilcoxon_p']}),
        ('post_test.json|P2.subset_minus_full', 'P2 AUC subset batches vs full-run batches, 23', {'higher': PT['P2']['subset_minus_full']['wins'], 'lower': PT['P2']['subset_minus_full']['losses'], 'p': PT['P2']['subset_minus_full']['wilcoxon_p']})]
for key in ['wss95_asr_jevblend_3_vs_asr_prior', 'wss100_asr_jevblend_3_vs_asr_prior', 'wss95_asr_jevblend_3_vs_asr_pseudo_10_50', 'wss100_asr_jevblend_3_vs_asr_pseudo_10_50']:
    met, rest = key.split('_', 1); a, b = rest.split('_vs_')
    ORIG.append((f'clef_results.json|{key}', f'CLEF {met} {a} vs {b}, 28 reviews', {'higher': CL[key]['wins'], 'p': CL[key]['p']}))
B13 = {r['key']: r for r in ALL if r['analysis'] == 'AN-0001-13'}
orig_rows = []
for src, name, st in ORIG:
    e = B13[name]
    row = {'source': src, 'test': name, 'stored_higher': st.get('higher'), 'stored_lower': st.get('lower'), 'stored_p': st.get('p'), 'exact_higher': e['higher'], 'exact_lower': e['lower'], 'exact_tied': e['tied'], 'exact_p': e['p_exact_arithmetic'], 'method': e.get('method')}
    row['counts_changed'] = (st.get('higher') is not None and st['higher'] != e['higher']) or (st.get('lower') is not None and st['lower'] != e['lower'])
    row['p_changed'] = st.get('p') is not None and abs(st['p'] - e['p_exact_arithmetic']) > 1e-12 * max(1.0, abs(st['p']))
    row['p_printed_changed'] = st.get('p') is not None and fmt_p(st['p']) != fmt_p(e['p_exact_arithmetic'])
    orig_rows.append(row)
OUT['values_stored_by_the_original_analysis'] = orig_rows
# C4 of the original analysis: reviews in which the threshold read fewer records than the knee method (held-out), quoted in table 2 of the manuscript
e4 = [r for r in ALL if r['analysis'] == 'AN-0001-04' and r['key'] == 'heldout|paired_threshold_vs_knee_asr|work'][0]
OUT['C4_heldout_threshold_reads_less_than_knee'] = {'exact_less': e4['lower'], 'exact_more': e4['higher'], 'exact_equal': e4['tied'], 'stored_less': e4['stored']['lower'], 'stored_more': e4['stored']['higher']}

# ------------------------------------------------------------------ 5. leave-one-review-out p values of AN-0001-08
LOO8 = R8['leave_one_review_out']
def loo_p(name, group, quantity, a_ex, b_ex, revs):
    d = np.array([float(x - y) for x, y in zip(a_ex, b_ex)])
    full, _ = wil(d); vals_ = []
    for i in range(len(d)):
        m = np.ones(len(d), bool); m[i] = False; p, _ = wil(d[m]); vals_.append(p)
    a = np.array(vals_, float); st = LOO8[group][quantity]
    r = {'quantity': quantity, 'group': group, 'full': full, 'loo_min': float(a.min()), 'review_min': revs[int(a.argmin())], 'loo_max': float(a.max()), 'review_max': revs[int(a.argmax())],
         'stored_full': st['full'], 'stored_loo_min': st['loo_min'], 'stored_review_min': st['review_min'], 'stored_loo_max': st['loo_max'], 'stored_review_max': st['review_max']}
    r['changed'] = any(abs(r[k] - r['stored_' + k]) > 1e-12 * max(1.0, abs(r[k])) for k in ['full', 'loo_min', 'loo_max']) or r['review_min'] != st['review_min'] or r['review_max'] != st['review_max']
    return r
hr = list(PR[PR.collection == 'heldout'].review); cr = list(PR[PR.collection == 'clef'].review); c1r = list(C1.index)
OUT['leave_one_review_out_p'] = [
    loo_p('C1', 'C1', 'C1_wilcoxon_p', [C1EX[k][0] for k in c1r], [C1EX[k][1] for k in c1r], c1r),
    loo_p('C3', 'C3_and_E1.04', 'C3_wilcoxon_p', [EX8['heldout'][k]['hybrid_wss95_stored'] for k in hr], [EX8['heldout'][k]['asr_wss95_stored'] for k in hr], hr),
    loo_p('hybrid minus Jev ranking', 'C3_and_E1.04', 'hybrid_minus_jev_only_wilcoxon_p', [EX8['heldout'][k]['hybrid_wss95_stored'] for k in hr], [EX8['heldout'][k]['jev_wss95'] for k in hr], hr),
    loo_p('CLEF hybrid minus ASReview', 'CLEF', 'CLEF_hybrid_minus_asr_wilcoxon_p', [EX8['clef'][k]['hybrid_wss95_stored'] for k in cr], [EX8['clef'][k]['asr_wss95_stored'] for k in cr], cr)]
# criteria C1 and C3 with one review removed: the decision rule of the plan (difference and p value), in exact arithmetic
def c_pass(a_ex, b_ex, min_diff, alpha):
    d = [x - y for x, y in zip(a_ex, b_ex)]; res = []
    for i in range(len(d)):
        dd = [v for j, v in enumerate(d) if j != i]; p, _ = wil(np.array([float(v) for v in dd])); res.append(bool(mean_f(dd) >= F(min_diff) and p < alpha))
    p, _ = wil(np.array([float(v) for v in d]))
    return {'all_reviews': bool(mean_f(d) >= F(min_diff) and p < alpha), 'with_one_review_removed_all_pass': all(res)}
OUT['criteria_decisions_exact_arithmetic'] = {'C1 (difference >= 0.03 and p < 0.01)': c_pass([C1EX[k][0] for k in c1r], [C1EX[k][1] for k in c1r], '3/100', 0.01),
                                              'C3 WSS@95 part (difference >= 0.05 and p < 0.05)': c_pass([EX8['heldout'][k]['hybrid_wss95_stored'] for k in hr], [EX8['heldout'][k]['asr_wss95_stored'] for k in hr], '5/100', 0.05)}

# ------------------------------------------------------------------ 6. AN-0001-01: reviews with a higher value with batches (tolerance rule only; model fits are not rational numbers)
A1 = pd.read_csv(AN + '/AN-0001-01/per_review_subset2002_final_single.csv').set_index('review'); B1 = pd.read_csv(AN + '/AN-0001-01/per_review_subset2002_final_b10_full.csv').set_index('review')
MD = J(AN + '/AN-0001-01/results.json')['subset2002_per_review_batched_minus_single']; conv = (A1.get('recal_converged') == True) & (B1.get('recal_converged') == True); rows01 = []
for c, e in MD.items():
    x = A1[c]; z = B1[c]; ok = x.notna() & z.notna()
    if c.startswith('recal_'): ok = ok & conv
    d = (z - x)[ok].values; dz = np.where(np.abs(d) <= TOL, 0.0, d)
    rows01.append({'quantity': c, 'n': int(ok.sum()), 'stored_n': e['n'], 'stored_higher_with_batches': e['n_b10_higher'], 'float_higher': int((d > 0).sum()), 'tol_higher': int((dz > 0).sum()), 'tol_lower': int((dz < 0).sum()), 'tol_tied': int((dz == 0).sum()),
                   'smallest_absolute_difference': float(np.min(np.abs(d))), 'changed': int((dz > 0).sum()) != e['n_b10_higher']})
OUT['AN-0001-01_scoring_modes'] = rows01

# ------------------------------------------------------------------ summary and output
def summ(an):
    S = [r for r in ALL if r['analysis'] == an]
    return {'comparisons': len(S), 'counts_changed': sum(1 for r in S if r.get('counts_changed')), 'p_changed': sum(1 for r in S if r.get('p_changed')),
            'p_changed_at_printed_precision': sum(1 for r in S if r.get('stored') and r['stored'].get('p') is not None and r.get('p_exact_arithmetic') is not None and fmt_p(r['stored']['p']) != fmt_p(r['p_exact_arithmetic'])),
            'method_changed': sum(1 for r in S if r.get('stored') and r['stored'].get('method') and r.get('method') and str(r['stored']['method']).split('_')[0] != str(r['method']).split(' ')[0]),
            'float_counts_not_equal_stored': sum(1 for r in S if r.get('float_counts_equal_stored') is False), 'float_p_not_equal_stored': sum(1 for r in S if r.get('float_p_equals_stored') is False),
            'exact_counts_differ_from_tolerance_rule': sum(1 for r in S if 'tol_higher' in r and (r['higher'], r['lower'], r['tied']) != (r['tol_higher'], r['tol_lower'], r['tol_tied'])),
            'p_exact_differs_from_tolerance_rule_at_printed_precision': sum(1 for r in S if r.get('p_tol') is not None and fmt_p(r['p_tol']) != fmt_p(r['p_exact_arithmetic']))}
OUT['summary'] = {an: summ(an) for an in ['AN-0001-02', 'AN-0001-03', 'AN-0001-04', 'AN-0001-08', 'AN-0001-13', 'original']}
OUT['gap_check_differences_between_1e-12_and_1e-7'] = [{'analysis': a, 'key': k, 'values': g} for a, k, g in GAP]
OUT['comparisons'] = ALL
OUT['finished'] = now()
json.dump(OUT, open(os.path.join(HERE, 'fx01_exact_paired.json'), 'w'), indent=1, default=str)
flat = []
for r in ALL:
    s = r.get('stored') or {}
    flat.append({'analysis': r['analysis'], 'key': r['key'], 'n': r['n'], 'stored_higher': s.get('higher'), 'stored_lower': s.get('lower'), 'stored_tied': s.get('tied'), 'stored_p': s.get('p'),
                 'exact_higher': r['higher'], 'exact_lower': r['lower'], 'exact_tied': r['tied'], 'tied_absolute_differences': r.get('tied_absolute_differences'), 'exact_p': r.get('p_exact_arithmetic'), 'method': r.get('method'),
                 'tolerance_rule_p': r.get('p_tol'), 'counts_changed': r.get('counts_changed'), 'p_changed': r.get('p_changed'), 'reviews_tied_but_counted_as_different': ';'.join(map(str, r.get('reviews_tied_but_counted_as_different', []))),
                 'stored_p_printed': fmt_p(s.get('p')), 'exact_p_printed': fmt_p(r.get('p_exact_arithmetic'))})
pd.DataFrame(flat).to_csv(os.path.join(HERE, 'fx01_exact_paired.csv'), index=False)
log(json.dumps(OUT['summary'], indent=1))
log('gap check (differences between 1e-12 and 1e-7):', len(GAP))
log('values stored by the original analysis that change:', json.dumps([r for r in orig_rows if r['counts_changed'] or r['p_changed']], indent=1))
log('leave-one-out p values:', json.dumps(OUT['leave_one_review_out_p'], indent=1))
log('criteria:', json.dumps(OUT['criteria_decisions_exact_arithmetic'], indent=1))
log('C4 reads less:', json.dumps(OUT['C4_heldout_threshold_reads_less_than_knee']))
log('AN-0001-01:', json.dumps(rows01, indent=1))
log('finished', OUT['finished'])
