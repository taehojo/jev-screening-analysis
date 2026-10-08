# AN-0001-05 (npj round 1): records changing side of tau=0.07 between the three Jev scoring modes
# of the 2002-record held-out subset. Post hoc. Reads stored outputs only; no model call.
import json, sys, datetime, itertools
import numpy as np
S = '/N/project/AiLab/jev/synergy/'
TAU = 0.07
ids = json.load(open(S + 'test_cmp_ids.json'))
idset = set(ids)
recs = {r['id']: r for r in json.load(open(S + 'test_records.json')) if r['id'] in idset}
modes = {'single': S + 'jev_testcmp_single.json', 'b10_subset': S + 'jev_testcmp_b10.json', 'b10_full': S + 'jev_test.json'}
P = {}
for m, fn in modes.items():
    d = {}
    for e in json.load(open(fn)):
        if e['id'] in idset:
            d[e['id']] = (e.get('p') if e.get('ok') else None)
    P[m] = d
out = {'analysis_id': 'AN-0001-05', 'tau': TAU, 'n_subset_ids': len(ids), 'n_records_found': len(recs),
       'inputs': {'ids': S + 'test_cmp_ids.json', 'records': S + 'test_records.json', **modes},
       'missing_or_failed': {m: sum(1 for i in ids if P[m].get(i) is None) for m in modes}}
common = [i for i in ids if all(P[m].get(i) is not None for m in modes)]
out['n_common'] = len(common)

def lab(i, level):
    r = recs[i]
    v = r['label'] if level == 'final' else r.get('label_ta')
    if v is None or v == '': return None
    return int(v)

for level in ('final', 'ta'):
    L = {}
    rows = [i for i in common if lab(i, level) is not None]
    revs = sorted({recs[i]['review'] for i in rows})
    res = {'n_records': len(rows), 'n_reviews': len(revs), 'n_positive': sum(lab(i, level) for i in rows)}
    # per mode 2x2
    ct = {}
    for m in modes:
        c = {'pos_below': 0, 'pos_at_or_above': 0, 'neg_below': 0, 'neg_at_or_above': 0}
        for i in rows:
            y = lab(i, level); above = P[m][i] >= TAU
            c[('pos' if y else 'neg') + ('_at_or_above' if above else '_below')] += 1
        ct[m] = c
    res['crosstab_by_mode'] = ct
    # paired side-of-tau agreement per pair of modes, by class
    pairs = {}
    for a, b in itertools.combinations(modes, 2):
        pr = {}
        for cls, cname in ((1, 'positive'), (0, 'negative'), (None, 'all')):
            sel = [i for i in rows if cls is None or lab(i, level) == cls]
            t = {'both_below': 0, 'both_at_or_above': 0, f'{a}_below_{b}_at_or_above': 0, f'{a}_at_or_above_{b}_below': 0}
            for i in sel:
                ba, bb = P[a][i] < TAU, P[b][i] < TAU
                if ba and bb: t['both_below'] += 1
                elif (not ba) and (not bb): t['both_at_or_above'] += 1
                elif ba: t[f'{a}_below_{b}_at_or_above'] += 1
                else: t[f'{a}_at_or_above_{b}_below'] += 1
            t['n'] = len(sel); t['discordant'] = t[f'{a}_below_{b}_at_or_above'] + t[f'{a}_at_or_above_{b}_below']
            pr[cname] = t
        pairs[f'{a}_vs_{b}'] = pr
    res['paired_side_of_tau'] = pairs
    # per review recall at tau per mode (within the subset)
    per = []
    for k in revs:
        rr = [i for i in rows if recs[i]['review'] == k]
        n1 = sum(lab(i, level) for i in rr)
        row = {'review': k, 'n_subset': len(rr), 'n_pos': n1}
        for m in modes:
            f = sum(1 for i in rr if lab(i, level) == 1 and P[m][i] >= TAU)
            row[f'recall_{m}'] = (f / n1) if n1 else None
            row[f'pos_below_{m}'] = n1 - f
        per.append(row)
    res['per_review'] = per
    res['reviews_recall_below_0.95'] = {m: [r['review'] for r in per if r[f'recall_{m}'] is not None and r[f'recall_{m}'] < 0.95] for m in modes}
    res['n_reviews_recall_below_0.95'] = {m: len(v) for m, v in res['reviews_recall_below_0.95'].items()}
    res['n_reviews_with_positive'] = sum(1 for r in per if r['n_pos'] > 0)
    res['min_recall'] = {m: min(r[f'recall_{m}'] for r in per if r[f'recall_{m}'] is not None) for m in modes}
    out[level] = res

out['note'] = ('The subset holds all 597 finally included records and 1405 excluded records sampled at about 4.3% per review '
               '(29.8% included), so proportions read are not workload estimates and are not reported. At the title-and-abstract level '
               '(12 reviews), positives in the subset are the finally included records plus the title-and-abstract positives that happened '
               'to be among the sampled excluded records, so recall at that level describes the subset only.')
out['finished'] = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
json.dump(out, open('results.json', 'w'), indent=1)
import csv
for level in ('final', 'ta'):
    with open(f'per_review_{level}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[level]['per_review'][0].keys())); w.writeheader(); w.writerows(out[level]['per_review'])
for level in ('final', 'ta'):
    r = out[level]
    print(level, 'records', r['n_records'], 'reviews', r['n_reviews'], 'positives', r['n_positive'])
    for m, c in r['crosstab_by_mode'].items(): print('  ', m, c)
    for p, v in r['paired_side_of_tau'].items(): print('  ', p, 'pos', v['positive'], 'neg', v['negative'])
    print('   reviews recall<0.95', r['reviews_recall_below_0.95'])
    print('   min recall', r['min_recall'])
print('missing', out['missing_or_failed'], 'common', out['n_common'])
