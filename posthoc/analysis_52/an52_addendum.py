#!/usr/bin/env python3
"""AN-0004-01 addendum (logged 2026-09-29 09:18:08 EDT before execution).
For each evaluated option and collection, compare the set of reviews judged not reliable on the
mean recall over runs (mean < 0.95) with the set of reviews having at least one run below 0.95
(runs_below_095_by_review in results.json), and count reviews that are reliable on the mean but
have at least one run below 0.95. Same stored inputs as an52_single_run.py; standard library only.
"""
import csv, json, os
from fractions import Fraction as F

OUT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(OUT, 'results.json')))
I = R['inputs']
rd = lambda p: list(csv.DictReader(open(p, newline='')))
an02, an11, l08 = rd(I['an02_per_review']), rd(I['an11_per_review']), rd(I['l08_per_order'])
AN11C = {'heldout': 'heldout_final', 'clef': 'clef28_content'}
out = {'analysis_id': 'AN-0004-01', 'part': 'addendum', 'logged': '2026-09-29 09:18:08 EDT', 'results': {}}
for coll in ('heldout', 'clef'):
    A02 = {r['review']: r for r in an02 if r['collection'] == coll}
    A11 = {r['review']: r for r in an11 if r['collection'] == AN11C[coll]}
    P = {}
    for r in l08:
        if r['collection'] == coll and r['method'] in ('asr_prior', 'jev_only'):
            P.setdefault((r['review'], r['method']), []).append(r)
    revs = sorted(A02)
    mean = {
        'threshold': {k: F(A02[k]['thr_rec']) for k in revs},
        'knee150_asr': {k: F(A02[k]['asr_knee150_rec']) for k in revs},
        'knee1000_asr': {k: F(A02[k]['asr_knee1000_rec']) for k in revs},
        'combined_workflow': {k: sum(F(x['stat_rec']) for x in P[(k, 'jev_only')]) / len(P[(k, 'jev_only')]) for k in revs},
        'stat_asreview': {k: sum(F(x['stat_rec']) for x in P[(k, 'asr_prior')]) / len(P[(k, 'asr_prior')]) for k in revs},
        'threshold_plus_sample_check': {k: F(A11[k]['check_rec_mean']) for k in revs},
    }
    res = {}
    for o, m in mean.items():
        opt = R['results'][coll]['options'][o]
        fail_mean = sorted(k for k in revs if m[k] < F(19, 20))
        below = opt['runs_below_095_by_review']
        any_below = sorted(below)
        hidden = sorted(k for k in any_below if k not in fail_mean)
        assert len(fail_mean) == len(revs) - opt['mean_based_reliable_reviews_table3'], (coll, o)
        res[o] = {
            'reviews_not_reliable_on_mean': fail_mean,
            'reviews_with_any_run_below_095': any_below,
            'sets_equal': fail_mean == any_below,
            'reviews_reliable_on_mean_with_any_run_below_095': hidden,
            'runs_below_095_in_reviews_reliable_on_mean': sum(below[k] for k in hidden),
            'runs_at_or_above_095_in_reviews_not_reliable_on_mean': sum(opt['runs_per_review'] - below.get(k, 0) for k in fail_mean),
            'mean_recall_of_reviews_not_reliable_on_mean': {k: float(m[k]) for k in fail_mean},
        }
        print(coll, o, json.dumps(res[o]), flush=True)
    out['results'][coll] = res
json.dump(out, open(os.path.join(OUT, 'results_addendum.json'), 'w'), indent=1)
print('wrote results_addendum.json', flush=True)
