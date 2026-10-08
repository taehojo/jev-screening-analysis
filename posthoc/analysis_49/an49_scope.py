# npj AN-0002-03 (manuscript analysis 49, post hoc): scope of the zero-difference check (analysis 39) and the same
# three tests applied to the printed Wilcoxon comparisons of analyses 32 to 46 that the check did not cover.
# (a) Maps the 383 comparisons of rounds/round_0001/revision/analysis/AN-0001-08/zero_sensitivity_all.csv to the v0001
#     Supplementary Tables through the table builders (npj_digital_medicine/v0003_copy/source/build/make_appendix_tables.py
#     for the tables carried over from v0, make_tables_v0001.py for analyses 32 to 46) and checks that the printed default
#     P value of each comparison appears in the assigned table or text of the v0001 SI (inputs/SI_v0001.md, converted from
#     versions/v0001_after_round0001/Supplementary_Information_v0001.docx with tools/docx2md.py).
# (b) Enumerates every comparison of analyses 32 to 46 whose Wilcoxon P value is printed (entries of
#     AUDITFIX_iter2_v0001/tables_trace_v0001.json whose expression ends in wilcoxon_p), rebuilds the per-review
#     differences from the stored per-review files, checks the default Wilcoxon P against the stored and the printed value,
#     identifies comparisons already among the 383 (same review set and identical differences up to sign), and applies to
#     the others the function tests() copied unchanged from AN-0001-08/an08_zero_sensitivity.py.
# Differences from an08_zero_sensitivity.py: new enumeration and input files; tests() is identical; outputs in this folder.
# No model call, no label read (only stored per-review metrics).
import json, os, re, math, warnings, subprocess
import numpy as np, pandas as pd, scipy
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__))
AN = '/N/project/AiLab/jev/review_pipeline_npj/rounds/round_0001/revision/analysis'
SEED = 20260928; BMC = 100000; MAXEXACT = 20
def now(): return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S %Z'], capture_output=True, text=True).stdout.strip()
T0 = now()

# ---------------------------------------------------------------- tests() copied unchanged from an08_zero_sensitivity.py
def auto_method(d):
    nz = d[d != 0]; ties = len(nz) - len(np.unique(np.abs(nz))); zeros = int((d == 0).sum())
    if zeros == 0 and ties == 0 and len(d) <= 50: return 'exact'
    if len(d) <= 13: return 'exact (permutation; zero or tied differences)'
    return 'asymptotic'
def tests(d):
    d = np.asarray(d, float); n = len(d); nz = d[d != 0]; m = len(nz); r = {'n': n, 'n_nonzero': m, 'n_zero': n - m, 'mean_diff': float(d.mean()) if n else None}
    if m == 0: r.update({'p_wilcoxon_default': None, 'p_pratt': None, 'p_signflip': 1.0, 'signflip_method': 'all differences zero'}); return r
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r['p_wilcoxon_default'] = float(wilcoxon(d).pvalue); r['wilcoxon_default_method'] = auto_method(d)
        try:
            r['p_pratt'] = float(wilcoxon(d, zero_method='pratt').pvalue)
        except Exception as e:
            r['p_pratt'] = None; r['pratt_error'] = str(e)[:120]
        try:
            r['p_pratt_asymptotic'] = float(wilcoxon(d, zero_method='pratt', method='asymptotic').pvalue)
        except Exception as e:
            r['p_pratt_asymptotic'] = None
        if r.get('p_pratt') is not None and r.get('p_pratt_asymptotic') is not None:
            r['pratt_auto_equals_asymptotic'] = abs(r['p_pratt'] - r['p_pratt_asymptotic']) <= 1e-15
    s_obs = abs(nz.sum()); tol = 1e-12 * max(1.0, np.abs(nz).sum())
    if m <= MAXEXACT:
        signs = 1.0 - 2.0 * ((np.arange(2 ** m, dtype=np.int64)[:, None] >> np.arange(m, dtype=np.int64)) & 1)
        S = np.abs(signs @ nz); r['p_signflip'] = float(np.mean(S >= s_obs - tol)); r['signflip_method'] = f'exact enumeration of 2^{m} sign patterns'
    else:
        rng = np.random.default_rng(SEED); signs = rng.choice([1.0, -1.0], size=(BMC, m))
        S = np.abs(signs @ nz); c = int(np.sum(S >= s_obs - tol)); r['p_signflip'] = float((1 + c) / (1 + BMC)); r['signflip_method'] = f'Monte Carlo, {BMC} random sign flips, seed {SEED}'
    return r

# ---------------------------------------------------------------- P formatting of make_tables_v0001.py (p and pn)
def pn(x):
    if x is None or (isinstance(x, float) and math.isnan(x)): return 'no test'
    if x < 0.0001: return '<0.0001'
    if x < 0.01: return '%#.2g' % x
    return f'{x:.3f}'
# P formatting of make_appendix_tables.py (pval), used by the tables carried over from v0, with '.' as decimal point
# (run 2: added after run 1 compared every carried-over table with the v0001 format and missed values printed with two decimals)
def pval_v0(p):
    if p is None or (isinstance(p, float) and math.isnan(p)): return 'no test'
    if p < 0.0001: return '<0.0001'
    if p >= 0.1: s = f'{p:.2f}'
    elif p >= 0.001:
        s = f'{p:.2g}'; s = f'{float(s):.{max(2, -int(math.floor(math.log10(p))) + 1)}f}'
    else:
        s = f'{float(f"{p:.2g}"):.5f}'.rstrip('0')
    return s

# ================================================================ (a) the 383 comparisons
Z = pd.read_csv(os.path.join(AN, 'AN-0001-08', 'zero_sensitivity_all.csv'))
NUM = json.load(open(os.path.join(AN, 'AUDITFIX_iter2_v0001', 'build', 'checks', 'numbering.json')))
TN, NN = NUM['tables'], NUM['notes']
SI = open(os.path.join(HERE, 'inputs', 'SI_v0001.md'), encoding='utf-8').read()
MS = open(os.path.join(HERE, 'inputs', 'Manuscript_v0001.md'), encoding='utf-8').read()
def si_block_table(n):
    i = SI.find(f'Supplementary Table {n}. ')
    if i < 0: return ''
    j = SI.find('**Supplementary Table ', i + 30); k = SI.find('Supplementary Note ', i + 30)
    ends = [x for x in (j, k) if x > 0]
    # the footnote of a table follows the table; take up to the next table header
    return SI[i: (j if j > 0 else len(SI))]
def si_note(n):
    m = re.search(rf'\n[#*\s]*Supplementary Note {n}\b[^\n]*\n', SI)
    if not m: return ''
    m2 = re.search(rf'\n[#*\s]*Supplementary Note {n + 1}\b[^\n]*\n', SI)
    return SI[m.start(): m2.start() if m2 else len(SI)]
NOTE_RECALL = next((n for n in range(1, 27) if 'Differences in recall at the stop' in si_note(n)), None)
NOTE_ATD = next((n for n in range(1, 27) if 'Paired differences in the average time to discovery' in si_note(n)), None)
def assign(src, key):
    if src == 'AN-0001-02': return ('table', TN['ranking_diffs'], 'ranking_diffs (built by make_appendix_tables.py; recount analysis AN-0001-02 of the earlier version)')
    if src == 'AN-0001-03': return ('table', TN['lambda_diffs'], 'lambda_diffs (make_appendix_tables.py; recount AN-0001-03)')
    if src == 'AN-0001-04': return ('table', TN['stopping_diffs'], 'stopping_diffs (make_appendix_tables.py; recount AN-0001-04)')
    if src == 'AN-0001-08':
        if key.endswith('_recall'): return ('note', NOTE_RECALL, 'stop_recall_diffs text (make_appendix_tables.py; recount AN-0001-08)')
        if '|atd_' in key: return ('note', NOTE_ATD, 'atd_diffs text (make_appendix_tables.py; recount AN-0001-08; no P printed there)')
        return ('table', TN['stopping_diffs'], 'stopping_diffs (make_appendix_tables.py; recount AN-0001-08)')
    if src == 'AN-0001-13': return ('table', TN['wilcoxon'], 'wilcoxon (make_appendix_tables.py; recount AN-0001-13)')
    if src == 'original': return ('search', None, 'development-review tests of the original analysis (not in a v0 table builder)')
    if src == 'AN-0001-02 (npj)': return ('table', TN['knee1000'], 'knee1000 (make_tables_v0001.py; analysis 33)')
    if src == 'AN-0001-03 (npj)': return ('table', TN['hybrid_seeds'], 'hybrid_seeds (make_tables_v0001.py; analysis 34)')
    return ('search', None, 'unassigned')
def src_is_original(s): return s == 'original'
Zmain = set(pd.read_csv(os.path.join(AN, 'AN-0001-08', 'zero_sensitivity_main.csv')).key)
rowsA = []
for _, r in Z.iterrows():
    kind, num, builder = assign(r['source'], r['key'])
    if kind == 'search' and src_is_original(r['source']):
        # run 3: the development-review results of the original analysis are printed in one paragraph of the SI
        i = SI.find('Development reviews (73). Jev macro-AUC'); txt = SI[i: SI.find('\n', i)] if i >= 0 else ''
    else:
        txt = si_block_table(num) if kind == 'table' else (si_note(num) if kind == 'note' else SI)
    pdef = r['p_wilcoxon_default']
    fmt = pval_v0 if 'make_appendix_tables' in builder or kind == 'search' else pn
    pr = fmt(pdef) if not (isinstance(pdef, float) and math.isnan(pdef)) else 'no test'
    found = (re.search(r'(?<![0-9.])' + re.escape(pr) + r'(?![0-9])', txt) is not None) if pr != 'no test' else None
    where_else = []
    if not found and pr != 'no test':
        rx = re.compile(r'(?<![0-9.])' + re.escape(pr) + r'(?![0-9])')
        for n in range(1, 60):
            if rx.search(si_block_table(n)): where_else.append(f'Table {n}')
        if rx.search(MS): where_else.append('manuscript')
    rowsA.append({'source': r['source'], 'key': r['key'], 'assigned': (f'Supplementary Table {num}' if kind == 'table' else (f'Supplementary Note {num}' if kind == 'note' else 'search')),
                  'builder': builder, 'in_table53_main_rows': r['key'] in Zmain, 'main_text_or_table2': r['main_text_or_table2'] if isinstance(r['main_text_or_table2'], str) else '',
                  'p_default_printed_format': pr, 'printed_p_found_in_assigned_text': found, 'p_string_found_elsewhere_in_SI_tables': ';'.join(where_else[:6]),
                  'n_zero': int(r['n_zero'])})
A = pd.DataFrame(rowsA); A.to_csv(os.path.join(HERE, 'scope_383_mapping.csv'), index=False)
countsA = A.groupby('assigned').size().to_dict()

# ================================================================ (b) printed Wilcoxon comparisons of analyses 32 to 46
TR = json.load(open(os.path.join(AN, 'AUDITFIX_iter2_v0001', 'tables_trace_v0001.json')))
RJ = {a: json.load(open(os.path.join(AN, a, 'results.json'))) for a in ['AN-0001-02', 'AN-0001-03', 'AN-0001-04', 'AN-0001-13']}
P2 = pd.read_csv(os.path.join(AN, 'AN-0001-02', 'per_review.csv'))
P3 = pd.read_csv(os.path.join(AN, 'AN-0001-03', 'per_review.csv'))
P4 = pd.read_csv(os.path.join(AN, 'AN-0001-04', 'per_review_tnr95.csv'))
P13 = pd.read_csv(os.path.join(AN, 'AN-0001-13', 'per_review_auc_tnr95.csv'))
def getp(obj, path):
    for k in path: obj = obj[k]
    return obj
STRATA = {'all': lambda T: T, 'N<=2000': lambda T: T[T.N <= 2000], 'N>2000': lambda T: T[T.N > 2000], 'N<1000': lambda T: T[T.N < 1000], 'N>=1000': lambda T: T[T.N >= 1000]}
def build(analysis, path):
    """per-review (reviews, a, b) for one printed comparison; path = key path in results.json ending in wilcoxon_p"""
    if analysis == 'AN-0001-02':            # summaries|coll|stratum|pair|work|wilcoxon_p
        _, coll, stratum, pair = path[:4]
        T = STRATA[stratum](P2[P2.collection == coll])
        a, b = {'paired_threshold_vs_knee1000_asr': ('thr_work', 'asr_knee1000_work'), 'paired_threshold_vs_knee150_asr': ('thr_work', 'asr_knee150_work'),
                'paired_knee1000_minus_knee150_asr': ('asr_knee1000_work', 'asr_knee150_work')}[pair]
        return list(T.review), T[a].values, T[b].values, f'{coll}|{stratum}|{pair}|work'
    if analysis == 'AN-0001-03':            # collections|coll|comparisons|name.metric|wilcoxon_p
        coll = path[1]; name, metric = path[3].split('.', 1)
        T = P3[P3.collection == coll]
        a, b = {'hybrid10_minus_asreview': ('hybrid10', 'asr10'), 'hybrid10_minus_jevonly': ('hybrid10', 'jevonly10'), 'jevonly_minus_asreview': ('jevonly10', 'asr10')}[name]
        return list(T.review), T[f'{a}_{metric}'].values, T[f'{b}_{metric}'].values, f'{coll}|{name}|{metric}'
    if analysis == 'AN-0001-04':            # collections|coll|comparisons|name|wilcoxon_p
        coll, name = path[1], path[3]
        T = P4[P4.collection == coll]
        a, b = {'hybrid_minus_asreview': ('tnr95_hybrid', 'tnr95_asreview'), 'jev_minus_asreview': ('tnr95_jev', 'tnr95_asreview'),
                'hybrid_minus_jev': ('tnr95_hybrid', 'tnr95_jev'), 'jev_minus_bge': ('tnr95_jev', 'tnr95_bge')}[name]
        return list(T.review), T[a].values, T[b].values, f'{coll}|tnr95|{name}'
    if analysis == 'AN-0001-13':            # sets|set|comparisons|name|wilcoxon_p  or sets|set|by_domain|dom|name|wilcoxon_p
        st = path[1]; T = P13[P13.set == st]
        if path[2] == 'by_domain':
            dom = path[3]; name = path[4]; T = T[T.biomedical == (dom == 'biomedical')]
        else:
            dom = None; name = path[3]
        a, b = {'auc_medcpt_minus_bge': ('auc_medcpt', 'auc_bge'), 'auc_jev_minus_medcpt': ('auc_jev', 'auc_medcpt')}[name]
        return list(T.review), T[a].values, T[b].values, f'{st}|' + (f'{dom}|' if dom else '') + name
    raise KeyError(analysis)

# difference vectors of the 383 (for coverage by identical data)
X = json.load(open(os.path.join(AN, 'AN-0001-08', 'fx01_exact_paired.json')))
VEC = {}
for c in X['comparisons']:
    if c.get('per_review_diff_float') is not None and c.get('reviews'):
        VEC[(c['analysis'], c['key'])] = (list(c['reviews']), np.array(c['per_review_diff_float'], float))
for coll in ('heldout', 'clef'):
    T = P2[P2.collection == coll]
    VEC[('AN-0001-02 (npj)', f'{coll}|threshold_minus_knee1000_asr|work')] = (list(T.review), (T.thr_work - T.asr_knee1000_work).values)
    VEC[('AN-0001-02 (npj)', f'{coll}|knee1000_minus_knee150_asr|work')] = (list(T.review), (T.asr_knee1000_work - T.asr_knee150_work).values)
    S = P3[P3.collection == coll]
    for met in ('wss95', 'wss100'):
        for a, b in [(f'hybrid10_{met}', f'asr10_{met}'), (f'hybrid10_{met}', f'jevonly10_{met}')]:
            VEC[('AN-0001-03 (npj)', f'{coll}|{a}_minus_{b}')] = (list(S.review), (S[a] - S[b]).values)
in383 = set(zip(Z.source, Z.key))
def covered(revs, d):
    hits = []
    order = np.argsort(revs); rs = [revs[i] for i in order]; ds = d[order]
    for k, (r2, d2) in VEC.items():
        if k not in in383 or len(r2) != len(revs): continue
        o2 = np.argsort(r2)
        if [r2[i] for i in o2] != rs: continue
        e2 = d2[o2]
        if np.max(np.abs(e2 - ds)) <= 1e-9 or np.max(np.abs(e2 + ds)) <= 1e-9: hits.append(f'{k[0]}: {k[1]}')
    return hits

TABLE_OF = {'knee1000': TN['knee1000'], 'hybrid_seeds': TN['hybrid_seeds'], 'tnr95': TN['tnr95'], 'medcpt': TN['medcpt']}
ANALYSIS_NO = {'AN-0001-02': 33, 'AN-0001-03': 34, 'AN-0001-04': 35, 'AN-0001-13': 44}
rowsB = []; mcnemar = []
for tname, L in TR.items():
    for e in L:
        m = re.match(r"J\('(AN-0001-\d\d)', '(.*)'\)$", e['expr'])
        if not m: continue
        an, path = m.group(1), m.group(2).split('|')
        if path[-1] in ('mcnemar_asymptotic_p', 'mcnemar_exact_p', 'mcnemar_midp'):
            mcnemar.append({'table': tname, 'expr': e['expr'], 'printed': e['printed']}); continue
        if path[-1] != 'wilcoxon_p': continue
        revs, a, b, label = build(an, path)
        d = np.asarray(a, float) - np.asarray(b, float)
        n_small = int(((d != 0) & (np.abs(d) <= 1e-12)).sum()); d = np.where(np.abs(d) <= 1e-12, 0.0, d)
        stored = getp(RJ[an], path)
        t = tests(d)
        mean_key = path[:-1] + ['mean_diff']
        try: stored_mean = getp(RJ[an], mean_key)
        except Exception: stored_mean = None
        cov = covered(revs, d)
        rowsB.append({'table': f'Supplementary Table {TABLE_OF[tname]}', 'table_key': tname, 'analysis': ANALYSIS_NO[an], 'result_file_key': f"{an}/results.json {'|'.join(path)}",
                      'label': label, 'n': len(d), 'n_zero': t['n_zero'], 'float_diffs_set_to_zero_(<=1e-12)': n_small,
                      'mean_diff': t['mean_diff'], 'stored_mean_diff': stored_mean,
                      'p_stored': stored, 'p_printed': e['printed'], 'p_default_recomputed': t['p_wilcoxon_default'],
                      'default_equals_stored_1e-12': (stored is not None and t['p_wilcoxon_default'] is not None and abs(stored - t['p_wilcoxon_default']) <= 1e-12 * max(1, abs(stored))),
                      'default_equals_printed': pn(t['p_wilcoxon_default']) == e['printed'],
                      'wilcoxon_default_method': t.get('wilcoxon_default_method'), 'p_pratt': t.get('p_pratt'), 'p_signflip': t.get('p_signflip'), 'signflip_method': t.get('signflip_method'),
                      'already_among_383': bool(cov), 'covered_by': '; '.join(cov)})
B = pd.DataFrame(rowsB)
B['sig05_default'] = B.p_default_recomputed < 0.05; B['sig05_pratt'] = B.p_pratt < 0.05; B['sig05_signflip'] = B.p_signflip < 0.05
B['conclusion_at_0.05_changes'] = (B.sig05_default != B.sig05_pratt) | (B.sig05_default != B.sig05_signflip)
B.to_csv(os.path.join(HERE, 'printed_wilcoxon_analyses32_46.csv'), index=False)
NEW = B[~B.already_among_383]
NEW.to_csv(os.path.join(HERE, 'zero_sensitivity_added.csv'), index=False)
res = {'analysis': 'AN-0002-03', 'manuscript_analysis_number': 49, 'post_hoc': True, 'started': T0,
       'environment': {'scipy': scipy.__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
       'part_a': {'n_comparisons': int(len(A)), 'by_assigned_location': {k: int(v) for k, v in countsA.items()},
                  'by_source': {k: int(v) for k, v in A.groupby('source').size().items()},
                  'n_printed_p_found_in_assigned_text': int((A.printed_p_found_in_assigned_text == True).sum()),
                  'n_printed_p_not_found_in_assigned_text': int((A.printed_p_found_in_assigned_text == False).sum()),
                  'not_found': A[A.printed_p_found_in_assigned_text == False][['source', 'key', 'assigned', 'p_default_printed_format', 'p_string_found_elsewhere_in_SI_tables']].to_dict('records'),
                  'n_in_table53_main_rows': int(A.in_table53_main_rows.sum()),
                  'note_numbers': {'stop_recall_diffs': NOTE_RECALL, 'atd_diffs': NOTE_ATD},
                  'analyses_32_46_among_383': {'analysis 33 (AN-0001-02 npj)': int((A.source == 'AN-0001-02 (npj)').sum()), 'analysis 34 (AN-0001-03 npj)': int((A.source == 'AN-0001-03 (npj)').sum())}},
       'part_b': {'n_printed_wilcoxon_comparisons_analyses_32_46': int(len(B)),
                  'by_table': {k: int(v) for k, v in B.groupby('table').size().items()},
                  'n_already_among_383': int(B.already_among_383.sum()), 'n_added': int(len(NEW)),
                  'added_by_table': {k: int(v) for k, v in NEW.groupby('table').size().items()},
                  'n_default_equals_stored': int(B.default_equals_stored_1e_12.sum()) if 'default_equals_stored_1e_12' in B else int(B['default_equals_stored_1e-12'].sum()),
                  'n_default_equals_printed': int(B.default_equals_printed.sum()),
                  'mismatches_default_vs_stored_or_printed': B[~(B['default_equals_stored_1e-12'] & B.default_equals_printed)][['table', 'label', 'p_stored', 'p_printed', 'p_default_recomputed']].to_dict('records'),
                  'n_float_diffs_set_to_zero_total': int(B['float_diffs_set_to_zero_(<=1e-12)'].sum()),
                  'added_n_with_at_least_one_zero_difference': int((NEW.n_zero > 0).sum()),
                  'added_n_conclusion_changes_at_0.05': int(NEW['conclusion_at_0.05_changes'].sum()),
                  'added_n_pratt_differs_at_0.05': int((NEW.sig05_default != NEW.sig05_pratt).sum()),
                  'added_n_signflip_differs_at_0.05': int((NEW.sig05_default != NEW.sig05_signflip).sum()),
                  'added_changes': NEW[NEW['conclusion_at_0.05_changes']][['table', 'label', 'n', 'n_zero', 'p_default_recomputed', 'p_pratt', 'p_signflip']].to_dict('records'),
                  'mcnemar_printed_entries_not_in_scope': len(mcnemar),
                  'mcnemar_comparisons_not_in_scope': len({re.sub(r'\|mcnemar_[a-z_]+\'\)$', '', x['expr']) for x in mcnemar})},
       'total_after_extension': {'comparisons': int(len(A) + len(NEW)), 'with_at_least_one_zero_difference': int((A.n_zero > 0).sum() + (NEW.n_zero > 0).sum())},
       'finished': None}
res['finished'] = now()
json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70); pd.set_option('display.max_rows', 200)
print(json.dumps({k: v for k, v in res['part_a'].items() if k != 'not_found'}, indent=0))
print('not found:', len(res['part_a']['not_found']))
for x in res['part_a']['not_found'][:40]: print('  ', x)
print(json.dumps({k: v for k, v in res['part_b'].items() if k not in ('added_changes', 'mismatches_default_vs_stored_or_printed')}, indent=0))
print('mismatches', res['part_b']['mismatches_default_vs_stored_or_printed'])
print(B[['table', 'label', 'n', 'n_zero', 'p_printed', 'p_default_recomputed', 'p_pratt', 'p_signflip', 'already_among_383', 'conclusion_at_0.05_changes']].to_string())
print('covered_by:'); print(B[B.already_among_383][['label', 'covered_by']].to_string())
