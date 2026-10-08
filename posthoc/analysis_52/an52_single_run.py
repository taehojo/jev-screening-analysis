#!/usr/bin/env python3
"""AN-0004-01 (manuscript analysis 52, post hoc): single-run reliability of the six evaluated
stopping options of Table 3 in the 23 held-out reviews (final inclusion) and the 28 Cochrane
reviews (CLEF 2019, content-level label).

Plan: review_pipeline_npj/REVISION_ANALYSIS_LOG.md, entry of 2026-09-29 09:07:49 EDT.
Reads stored per-run (or per-review per-run proportion) outputs only; no model call, no new
simulation, no network, no random step. Standard library only (csv, json, fractions, math).
A run reaches 95% recall when the number of included records found k satisfies
k >= ceil(0.95 * n1), computed with integers. Any failed consistency check stops the script
before results.json is written (the failure is printed to the log).
"""
import csv, json, math, os, sys, datetime
from fractions import Fraction as F

NPJ = '/N/project/AiLab/jev/review_pipeline_npj'
R1N = NPJ + '/rounds/round_0001/revision/analysis'
L8 = '/N/project/AiLab/jev/review_pipeline/rounds/round_0001/revision/analysis/AN-0001-08'
TAB3 = NPJ + '/versions/v0003_after_round0003/source/tables/table3_stopping.md'
OUT = os.path.dirname(os.path.abspath(__file__))
INPUTS = {
    'an02_per_review': R1N + '/AN-0001-02/per_review.csv',
    'an02_results': R1N + '/AN-0001-02/results.json',
    'an11_per_review': R1N + '/AN-0001-11/per_review.csv',
    'an11_results': R1N + '/AN-0001-11/results.json',
    'l08_per_order': L8 + '/per_order_stopping.csv',
    'l08_results': L8 + '/results.json',
    'table3': TAB3,
}
COLL = {'heldout': {'an11': 'heldout_final', 'n': 23}, 'clef': {'an11': 'clef28_content', 'n': 28}}
SEEDS = list(range(10))
TOL = 1e-9
checks = []


def fail(msg):
    print('CHECK FAILED:', msg, flush=True)
    print('Stopping before any output file is written.', flush=True)
    sys.exit(2)


def check(name, ok, detail):
    checks.append({'check': name, 'passed': bool(ok), 'detail': detail})
    print(('PASS ' if ok else 'FAIL ') + name + ' | ' + json.dumps(detail, default=str), flush=True)
    if not ok:
        fail(name)


def need(n1):
    return math.ceil(F(95, 100) * n1)


def to_k(r_str, n1, where):
    """Recall string -> integer found k, with |r - k/n1| < 1e-9."""
    if r_str is None or r_str.strip() == '' or r_str.strip().lower() == 'nan':
        fail(f'missing recall value at {where}')
    r = F(r_str)
    k = round(r * n1)
    if abs(float(r) - k / n1) >= TOL:
        fail(f'recall {r_str} is not k/n1 within 1e-9 at {where} (n1={n1})')
    return k


def read_csv(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


# ---------- load stored outputs ----------
an02 = read_csv(INPUTS['an02_per_review'])
an11 = read_csv(INPUTS['an11_per_review'])
l08 = read_csv(INPUTS['l08_per_order'])
an02r = json.load(open(INPUTS['an02_results']))
an11r = json.load(open(INPUTS['an11_results']))
l08r = json.load(open(INPUTS['l08_results']))

# Table 3 counts (mean-based reliable reviews), parsed from the v0003 generated table
tab3_rows = {}
for line in open(TAB3, encoding='utf-8'):
    if not line.startswith('| ') or line.startswith('| Stopping option') or line.startswith('|---'):
        continue
    cells = [c.strip() for c in line.strip().strip('|').split('|')]
    if len(cells) < 5 or ' of ' not in cells[3]:
        continue
    def cnt(c):
        a, b = c.split(';')[0].split(' of ')
        return int(a), int(b.split()[0])
    tab3_rows[cells[0]] = {'heldout': cnt(cells[3]), 'clef': cnt(cells[4])}
OPT_ROW = {
    'threshold': 'Threshold τ=0.07 on the Jev score',
    'knee150_asr': 'Knee method as adapted (from 150 screened records), default ASReview ranking',
    'knee1000_asr': 'Knee method with the published minimum of 1000 screened records, default ASReview ranking',
    'combined_workflow': 'Label-free Jev ranking plus statistical criterion (combined workflow)',
    'stat_asreview': 'Default ASReview ranking plus statistical criterion',
    'threshold_plus_sample_check': 'Threshold plus random-sample check below τ',
}
for o, rn in OPT_ROW.items():
    if rn not in tab3_rows:
        fail(f'Table 3 row not found: {rn}')
OPT_LABEL = {
    'threshold': 'Threshold tau=0.07 on the Jev score',
    'knee150_asr': 'Knee method as adapted (from 150 screened records), default ASReview ranking',
    'knee1000_asr': 'Knee method with the published minimum of 1000 screened records, default ASReview ranking',
    'combined_workflow': 'Label-free Jev ranking plus statistical criterion (combined workflow)',
    'stat_asreview': 'Default ASReview ranking plus statistical criterion',
    'threshold_plus_sample_check': 'Threshold plus random-sample check below tau',
}

results = {}
for coll, cinfo in COLL.items():
    A02 = {r['review']: r for r in an02 if r['collection'] == coll}
    A11 = {r['review']: r for r in an11 if r['collection'] == cinfo['an11']}
    P = {}
    for r in l08:
        if r['collection'] == coll and r['method'] in ('asr_prior', 'jev_only'):
            key = (r['review'], r['method'], int(r['seed']))
            if key in P:
                fail(f'duplicate per-order row {key}')
            P[key] = r
    revs = sorted(A02)
    check(f'{coll}: number of reviews', len(revs) == cinfo['n'] and sorted(A11) == revs,
          {'an02': len(revs), 'an11': len(A11), 'expected': cinfo['n'], 'same_set': sorted(A11) == revs})
    n1 = {k: int(A02[k]['n1']) for k in revs}
    check(f'{coll}: n1 agrees between AN-0001-02 and AN-0001-11', all(int(A11[k]['n1']) == n1[k] for k in revs), {})
    for meth in ('asr_prior', 'jev_only'):
        missing = [(k, s) for k in revs for s in SEEDS if (k, meth, s) not in P]
        extra = sorted({k for (k, m, s) in P if m == meth} - set(revs))
        check(f'{coll}: per-order rows {meth} seeds 0-9 present for every review', not missing and not extra,
              {'missing': missing[:5], 'n_missing': len(missing), 'extra_reviews': extra})
        badn1 = [k for k in revs for s in SEEDS if int(P[(k, meth, s)]['n1']) != n1[k]]
        check(f'{coll}: n1 agrees in per-order rows {meth}', not badn1, {'n_bad': len(badn1)})

    opt = {}
    # 1. threshold (deterministic; one run per review)
    runs = {}
    for k in revs:
        kk = to_k(A02[k]['thr_rec'], n1[k], f'{coll}/{k}/thr_rec')
        if kk != int(A02[k]['thr_found']):
            fail(f'thr_found differs from thr_rec at {coll}/{k}')
        k11 = to_k(A11[k]['thr_rec'], n1[k], f'{coll}/{k}/an11 thr_rec')
        if k11 != kk:
            fail(f'threshold recall differs between AN-0001-02 and AN-0001-11 at {coll}/{k}')
        runs[k] = [kk]
    opt['threshold'] = {'runs': runs, 'mean_based_source': 'thr_rec'}
    s = an11r[cinfo['an11']]['threshold_alone']
    mean_rel = sum(1 for k in revs if F(runs[k][0], n1[k]) >= F(19, 20))
    mean_rec = sum(runs[k][0] / n1[k] for k in revs) / len(revs)
    check(f'{coll}: threshold vs AN-0001-11 threshold_alone',
          mean_rel == s['n_reliable'] and abs(mean_rec - s['mean_recall']) < TOL,
          {'n_reliable': [mean_rel, s['n_reliable']], 'mean_recall': [mean_rec, s['mean_recall']]})

    # 2. knee as adapted (from 150), default ASReview ranking; 5. ASReview + statistical criterion
    for oname, meth, col in (('knee150_asr', 'asr_prior', 'knee_rec'), ('stat_asreview', 'asr_prior', 'stat_rec'),
                             ('combined_workflow', 'jev_only', 'stat_rec')):
        runs = {k: [to_k(P[(k, meth, sd)][col], n1[k], f'{coll}/{k}/{meth}/{sd}/{col}') for sd in SEEDS] for k in revs}
        opt[oname] = {'runs': runs}
    # checks for knee150
    r = opt['knee150_asr']['runs']
    diffs = [abs(sum(F(x, n1[k]) for x in r[k]) / 10 - F(A02[k]['asr_knee150_rec'])) for k in revs]
    check(f'{coll}: knee150 seed means equal AN-0001-02 asr_knee150_rec', max(diffs) < TOL, {'max_abs_diff': float(max(diffs))})
    mean_rel = sum(1 for k in revs if sum(F(x, n1[k]) for x in r[k]) / 10 >= F(19, 20))
    s = an02r['summaries'][coll]['all']['knee150_asr']
    check(f'{coll}: knee150 mean-based reliable reviews equal Table 3 and AN-0001-02',
          mean_rel == tab3_rows[OPT_ROW['knee150_asr']][coll][0] == s['n_reliable'],
          {'computed': mean_rel, 'table3': tab3_rows[OPT_ROW['knee150_asr']][coll], 'an02': s['n_reliable']})
    # checks for stat_asreview
    r = opt['stat_asreview']['runs']
    s = l08r['collections'][coll]['rules']['stat_asreview']
    mean_rel = sum(1 for k in revs if sum(F(x, n1[k]) for x in r[k]) / 10 >= F(19, 20))
    per_seed_mean = sum(F(sum(1 for x in r[k] if x >= need(n1[k])), 10) for k in revs) / len(revs)
    check(f'{coll}: stat_asreview vs Lancet AN-0001-08 rules.stat_asreview and Table 3',
          mean_rel == s['reliable_reviews'] == tab3_rows[OPT_ROW['stat_asreview']][coll][0]
          and abs(float(per_seed_mean) - s['reliability_per_seed_mean']) < TOL,
          {'reliable': [mean_rel, s['reliable_reviews'], tab3_rows[OPT_ROW['stat_asreview']][coll][0]],
           'reliability_per_seed_mean': [float(per_seed_mean), s['reliability_per_seed_mean']]})
    # checks for combined workflow
    r = opt['combined_workflow']['runs']
    s = an11r[cinfo['an11']]['combined_workflow_stored']
    mean_rel = sum(1 for k in revs if sum(F(x, n1[k]) for x in r[k]) / 10 >= F(19, 20))
    mean_rec = float(sum(sum(F(x, n1[k]) for x in r[k]) / 10 for k in revs) / len(revs))
    d_rev = max(abs(sum(F(x, n1[k]) for x in r[k]) / 10 - F(A11[k]['comb_rec_stored'])) for k in revs)
    check(f'{coll}: combined workflow vs AN-0001-11 combined_workflow_stored and Table 3',
          mean_rel == s['n_reliable'] == tab3_rows[OPT_ROW['combined_workflow']][coll][0] and abs(mean_rec - s['mean_recall']) < TOL,
          {'n_reliable': [mean_rel, s['n_reliable'], tab3_rows[OPT_ROW['combined_workflow']][coll][0]],
           'mean_recall': [mean_rec, s['mean_recall']], 'max_abs_diff_review_mean_vs_comb_rec_stored': float(d_rev)})

    # 3. knee with the published minimum of 1000: only the per-review proportion of seeds is stored
    cnt = {}
    for k in revs:
        v = F(A02[k]['asr_knee1000_rel_perseed']) * 10
        if abs(float(v) - round(v)) >= TOL:
            fail(f'asr_knee1000_rel_perseed x 10 is not an integer at {coll}/{k}')
        cnt[k] = int(round(v))
    opt['knee1000_asr'] = {'count_per_review': cnt}
    mean_rel = sum(1 for k in revs if F(A02[k]['asr_knee1000_rec']) >= F(19, 20))
    s = an02r['summaries'][coll]['all']
    psm = sum(F(A02[k]['asr_knee1000_rel_perseed']) for k in revs) / len(revs)
    check(f'{coll}: knee1000 mean-based reliable reviews equal Table 3 and AN-0001-02; per-seed mean equals summary',
          mean_rel == tab3_rows[OPT_ROW['knee1000_asr']][coll][0] == s['knee1000_asr']['n_reliable']
          and abs(float(psm) - s['asr_knee1000_reliability_per_seed_mean']) < TOL,
          {'reliable': [mean_rel, tab3_rows[OPT_ROW['knee1000_asr']][coll][0], s['knee1000_asr']['n_reliable']],
           'per_seed_mean': [float(psm), s['asr_knee1000_reliability_per_seed_mean']]})

    # 6. threshold plus random-sample check: per-review proportion of seeds and minimum over seeds
    cnt, mins = {}, {}
    for k in revs:
        v = F(A11[k]['check_rel_perseed']) * 10
        if abs(float(v) - round(v)) >= TOL:
            fail(f'check_rel_perseed x 10 is not an integer at {coll}/{k}')
        cnt[k] = int(round(v))
        mins[k] = to_k(A11[k]['check_rec_min_over_seeds'], n1[k], f'{coll}/{k}/check_rec_min_over_seeds')
        # consistency: a review with all ten runs at >= 95% must have its minimum at >= 95%, and conversely
        if (cnt[k] == 10) != (mins[k] >= need(n1[k])):
            fail(f'check_rel_perseed and check_rec_min_over_seeds disagree at {coll}/{k}')
    opt['threshold_plus_sample_check'] = {'count_per_review': cnt, 'min_k': mins}
    s = an11r[cinfo['an11']]['threshold_plus_sample_check']
    psm = sum(F(A11[k]['check_rel_perseed']) for k in revs) / len(revs)
    check(f'{coll}: sample check mean-based reliable reviews equal Table 3; per-seed mean equals AN-0001-11',
          s['n_reliable'] == tab3_rows[OPT_ROW['threshold_plus_sample_check']][coll][0]
          and abs(float(psm) - s['reliability_per_seed_mean']) < TOL,
          {'reliable': [s['n_reliable'], tab3_rows[OPT_ROW['threshold_plus_sample_check']][coll][0]],
           'per_seed_mean': [float(psm), s['reliability_per_seed_mean']]})
    # threshold Table 3 count
    check(f'{coll}: threshold reliable reviews equal Table 3',
          sum(1 for k in revs if opt['threshold']['runs'][k][0] >= need(n1[k])) == tab3_rows[OPT_ROW['threshold']][coll][0],
          {'table3': tab3_rows[OPT_ROW['threshold']][coll]})

    # ---------- metrics ----------
    out = {}
    for oname in OPT_ROW:
        o = opt[oname]
        if 'runs' in o:
            runs = o['runs']
            nrun = {k: len(runs[k]) for k in revs}
            ok = {k: sum(1 for x in runs[k] if x >= need(n1[k])) for k in revs}
            lowest = min((F(x, n1[k]), k) for k in revs for x in runs[k])
            low = {'value': float(lowest[0]), 'fraction': f'{lowest[0].numerator}/{lowest[0].denominator}', 'review': lowest[1],
                   'reviews_at_lowest': sorted({k for k in revs for x in runs[k] if F(x, n1[k]) == lowest[0]})}
            mean_based = sum(1 for k in revs if sum(F(x, n1[k]) for x in runs[k]) / len(runs[k]) >= F(19, 20))
        else:
            nrun = {k: 10 for k in revs}
            ok = dict(o['count_per_review'])
            if 'min_k' in o:
                lowest = min((F(o['min_k'][k], n1[k]), k) for k in revs)
                low = {'value': float(lowest[0]), 'fraction': f'{lowest[0].numerator}/{lowest[0].denominator}', 'review': lowest[1],
                       'reviews_at_lowest': sorted({k for k in revs if F(o['min_k'][k], n1[k]) == lowest[0]})}
            else:
                low = 'not stored'
            mean_based = tab3_rows[OPT_ROW[oname]][coll][0]
        N_pairs = sum(nrun.values()); N_ok = sum(ok.values())
        below = {k: nrun[k] - ok[k] for k in revs if ok[k] < nrun[k]}
        out[oname] = {
            'label': OPT_LABEL[oname],
            'runs_per_review': 1 if oname == 'threshold' else 10,
            'run_unit': {'threshold': 'deterministic rule, one run per review',
                         'knee150_asr': 'ASReview simulation seeds 0-9 (random prior records)',
                         'knee1000_asr': 'ASReview simulation seeds 0-9 (random prior records)',
                         'stat_asreview': 'ASReview simulation seeds 0-9 (random prior records)',
                         'combined_workflow': 'tie-breaking seeds 0-9 of the label-free Jev ranking',
                         'threshold_plus_sample_check': 'random-order seeds 0-9 of the sample below tau'}[oname],
            'n_reviews': len(revs),
            'review_run_pairs': N_pairs,
            'pairs_recall_ge_095': N_ok,
            'proportion_pairs_recall_ge_095': N_ok / N_pairs,
            'proportion_fraction': f'{N_ok}/{N_pairs}',
            'reviews_with_at_least_one_run_below_095': len(below),
            'runs_below_095_by_review': dict(sorted(below.items())),
            'lowest_single_run_recall': low,
            'mean_based_reliable_reviews_table3': tab3_rows[OPT_ROW[oname]][coll][0],
            'mean_based_reliable_reviews_recomputed': mean_based,
        }
    props = {o: F(out[o]['pairs_recall_ge_095'], out[o]['review_run_pairs']) for o in out}
    pmin = min(props.values())
    results[coll] = {'collection_label': {'heldout': '23 held-out SYNERGY+ reviews, final inclusion',
                                          'clef': '28 CLEF 2019 Cochrane reviews with a content-relevant record, content-level label'}[coll],
                     'options': out,
                     'lowest_proportion': {'value': float(pmin), 'fraction': f'{pmin.numerator}/{pmin.denominator}',
                                           'options': [o for o in out if props[o] == pmin]},
                     'all_options_every_pair_ge_095': all(p == 1 for p in props.values())}

res = {
    'analysis_id': 'AN-0004-01', 'manuscript_analysis': 52, 'post_hoc': True,
    'plan': 'REVISION_ANALYSIS_LOG.md entry of 2026-09-29 09:07:49 EDT',
    'definition': 'A review-run pair reaches 95% recall when the number of included records found k >= ceil(0.95*n1) (integer comparison).',
    'descriptive_reference_0.03': 'not computed: deterministic rule with recall 100% in every review of both collections by construction (Table 3)',
    'no_interval_or_test': 'runs are nested within reviews; descriptive counts only',
    'inputs': INPUTS,
    'consistency_checks': checks,
    'results': results,
    'finished': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
}
json.dump(res, open(os.path.join(OUT, 'results.json'), 'w'), indent=1, ensure_ascii=False)
with open(os.path.join(OUT, 'single_run_table.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['collection', 'option', 'runs_per_review', 'review_run_pairs', 'pairs_recall_ge_095', 'proportion',
                'reviews_with_at_least_one_run_below_095', 'lowest_single_run_recall', 'mean_based_reliable_reviews_table3', 'n_reviews'])
    for coll in results:
        for o, v in results[coll]['options'].items():
            low = v['lowest_single_run_recall']
            w.writerow([coll, o, v['runs_per_review'], v['review_run_pairs'], v['pairs_recall_ge_095'], f"{v['proportion_pairs_recall_ge_095']:.6f}",
                        v['reviews_with_at_least_one_run_below_095'], low if isinstance(low, str) else f"{low['value']:.6f}",
                        v['mean_based_reliable_reviews_table3'], v['n_reviews']])
print('all consistency checks passed; wrote results.json and single_run_table.csv', flush=True)
