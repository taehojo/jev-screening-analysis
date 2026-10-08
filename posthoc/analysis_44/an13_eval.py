#!/usr/bin/env python
# npj AN-0001-13 step 3 (post hoc): AUC and TNR@95 of the MedCPT Cross-Encoder against Jev and bge-base.
# Held-out 23 reviews (final inclusion), CLEF 28 reviews (content-level label) and 31 reviews (title and abstract label).
# Domain of held-out reviews: OpenAlex primary_topic domain of the review publication (metadata_publication.json of the SYNERGY+
# release copy read as a file): Health Sciences or Life Sciences = biomedical; Social Sciences or Physical Sciences = other.
import os, json, math, time, subprocess, platform, warnings
import numpy as np, pandas as pd, scipy, sklearn
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = '/N/project/AiLab/jev'; SYN = ROOT + '/synergy'
SRC = '/N/u/tjo/Quartz/.synergy_dataset_source/synergy-dataset-plus'
B = 5000; BOOT_SEED = 20260928; SEEDS = list(range(10))
def now(): return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S %Z'], capture_output=True, text=True).stdout.strip()
def J(p): return json.load(open(p))
def boot_ci(d):
    d = np.asarray(d, float); rng = np.random.default_rng(BOOT_SEED); m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
def method_rule(d):
    nz = d[d != 0]; ties = len(nz) - len(np.unique(np.abs(nz))); zeros = int((d == 0).sum())
    if zeros == 0 and ties == 0 and len(d) <= 50: return 'exact'
    if len(d) <= 13: return 'exact (permutation; zero or tied differences)'
    return 'asymptotic'
def paired(a, b, label):
    a = np.asarray(a, float); b = np.asarray(b, float); d = a - b
    out = {'label': label, 'n': len(d), 'mean_a': float(a.mean()), 'mean_b': float(b.mean()), 'mean_diff': float(d.mean()), 'ci95_bootstrap': boot_ci(d) if len(d) > 1 else None,
           'higher': int((d > 0).sum()), 'lower': int((d < 0).sum()), 'tied': int((d == 0).sum()), 'wilcoxon_p': None, 'wilcoxon_method': None}
    if len(d) > 5 and np.any(d != 0):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); out['wilcoxon_p'] = float(wilcoxon(d).pvalue)
        out['wilcoxon_method'] = method_rule(d)
    return out
def tnr95(seq, y):
    ys = y[np.asarray(seq)]; N = len(y); n1 = int(y.sum()); n0 = N - n1; need = math.ceil(0.95 * n1)
    k95 = int((np.where(ys == 1)[0] + 1)[need - 1]); return (n0 - (k95 - need)) / n0
def tnr95_mean(s, y):
    N = len(y); return float(np.mean([tnr95(np.lexsort((np.random.default_rng(t).random(N), -s)), y) for t in SEEDS]))

def domain(k):
    p = f'{SRC}/{k}/metadata_publication.json'
    if not os.path.exists(p): return None, None
    pt = J(p).get('primary_topic') or {}
    dom = (pt.get('domain') or {}).get('display_name'); fld = (pt.get('field') or {}).get('display_name'); return dom, fld

def main():
    t0 = time.time(); res = {'analysis_id': 'AN-0001-13 (npj round 1)', 'post_hoc': True, 'started': now(), 'model': J(os.path.join(HERE, 'model_download.json'))['revision_sha'],
                             'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
                             'settings': {'auc': 'sklearn roc_auc_score per review', 'tnr95': 'mean over ten random tie orders (seeds 0-9)', 'bootstrap': f'percentile, {B} resamples of reviews, seed {BOOT_SEED}'},
                             'sets': {}}
    rows = []
    for setname, rp, jp, zs, lab in [('heldout_final', SYN + '/test_records.json', SYN + '/jev_test.json', 'test', 'label'),
                                     ('clef_content', ROOT + '/clef/clef_records.json', SYN + '/jev_clef.json', 'clef', 'label'),
                                     ('clef_ta', ROOT + '/clef/clef_records.json', SYN + '/jev_clef.json', 'clef', 'label_ta')]:
        by = {}
        for r in J(rp): by.setdefault(r['review'], []).append(r)
        P = {d['id']: d['p'] for d in J(jp) if d.get('ok') and d.get('p') is not None}
        T = []; trunc = 0; npairs = 0
        for k in sorted(by):
            rs = by[k]
            if any(r.get(lab) is None for r in rs) and lab == 'label_ta' and setname.startswith('heldout'): continue
            y = np.array([int(r[lab] or 0) for r in rs])   # as synergy/clef_eval.py (r[lab] or 0)
            if not (0 < y.sum() < len(y)) or not all(r['id'] in P for r in rs): continue
            S = J(os.path.join(HERE, 'scores', f'{zs}_{k}.json')); assert S['ids'] == [r['id'] for r in rs]
            m = np.array(S['medcpt_ce'], float); jv = np.array([float(P[r['id']]) for r in rs]); bg = np.array(J(f'{SYN}/scores_zs/{zs}_{k}.json')['bge'], float)
            trunc += S['n_pairs_truncated']; npairs += len(rs)
            dom, fld = domain(k) if zs == 'test' else ('Health Sciences (Cochrane)', 'Medicine')
            T.append({'set': setname, 'review': k, 'N': len(y), 'n1': int(y.sum()), 'domain': dom, 'field': fld,
                      'auc_medcpt': roc_auc_score(y, m), 'auc_jev': roc_auc_score(y, jv), 'auc_bge': roc_auc_score(y, bg),
                      'tnr95_medcpt': tnr95_mean(m, y), 'tnr95_jev': tnr95_mean(jv, y), 'tnr95_bge': tnr95_mean(bg, y)})
        T = pd.DataFrame(T); rows.append(T)
        R = {'n_reviews': len(T), 'records': int(T.N.sum()), 'pairs_truncated_at_512': trunc, 'pairs': npairs,
             'macro': {c: float(T[c].mean()) for c in ['auc_medcpt', 'auc_jev', 'auc_bge', 'tnr95_medcpt', 'tnr95_jev', 'tnr95_bge']},
             'comparisons': {'auc_jev_minus_medcpt': paired(T.auc_jev, T.auc_medcpt, 'AUC Jev minus MedCPT Cross-Encoder'),
                             'auc_medcpt_minus_bge': paired(T.auc_medcpt, T.auc_bge, 'AUC MedCPT Cross-Encoder minus bge-base'),
                             'tnr95_jev_minus_medcpt': paired(T.tnr95_jev, T.tnr95_medcpt, 'TNR@95 Jev minus MedCPT Cross-Encoder'),
                             'tnr95_medcpt_minus_bge': paired(T.tnr95_medcpt, T.tnr95_bge, 'TNR@95 MedCPT Cross-Encoder minus bge-base')}}
        if setname == 'heldout_final':
            T['biomedical'] = T.domain.isin(['Health Sciences', 'Life Sciences'])
            R['by_domain'] = {}
            for flag, g in T.groupby('biomedical'):
                R['by_domain']['biomedical' if flag else 'other'] = {'n_reviews': len(g), 'reviews': g.review.tolist(), 'fields': g.field.value_counts().to_dict(),
                                                                     'macro': {c: float(g[c].mean()) for c in ['auc_medcpt', 'auc_jev', 'auc_bge']},
                                                                     'auc_jev_minus_medcpt': paired(g.auc_jev, g.auc_medcpt, 'AUC Jev minus MedCPT'),
                                                                     'auc_medcpt_minus_bge': paired(g.auc_medcpt, g.auc_bge, 'AUC MedCPT minus bge-base')}
        res['sets'][setname] = R
        print(setname, R['n_reviews'], {c: round(v, 4) for c, v in R['macro'].items()}, flush=True)
        for kk, v in R['comparisons'].items(): print(f"  {kk}: {v['mean_diff']:.4f} {v['ci95_bootstrap']} {v['higher']}/{v['lower']}/{v['tied']} p={v['wilcoxon_p']}", flush=True)
    # cross-checks with stored AUCs
    zt = pd.read_csv(SYN + '/zs_auc_test.csv').set_index('review'); T0 = rows[0].set_index('review')
    cz = pd.read_csv(SYN + '/clef_zs_label.csv').set_index('topic'); T1 = rows[1].set_index('review')
    res['cross_checks'] = {'heldout_bge_auc_max_abs_diff_vs_zs_auc_test_csv': float((T0.auc_bge - zt.loc[T0.index, 'bge']).abs().max()),
                           'clef_jev_auc_max_abs_diff_vs_clef_zs_label_csv': float((T1.auc_jev - cz.loc[T1.index, 'jev_AUC']).abs().max()),
                           'clef_bge_auc_max_abs_diff_vs_clef_zs_label_csv': float((T1.auc_bge - cz.loc[T1.index, 'bge_AUC']).abs().max())}
    print('cross checks', res['cross_checks'], flush=True)
    pd.concat(rows).to_csv(os.path.join(HERE, 'per_review_auc_tnr95.csv'), index=False)
    res['finished'] = now(); res['elapsed_seconds'] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'), indent=1, default=float)

if __name__ == '__main__':
    main()
