# AN-0001-19: which CLEF 2019 relevance labels do the LGAR (Jaumann et al 2025) Table 1 inclusion rates match?
# Reads only public CLEF qrels (clef/tar) and the project's clef_records.json; writes results.json in this folder.
import glob, collections, json, os, subprocess
ROOT = '/N/project/AiLab/jev'
LGAR_TABLE1 = {'DTA': {'n_slr': 8, 'papers': 30521, 'avg_incl_rate_pct': 7.05}, 'Intervention': {'n_slr': 20, 'papers': 41996, 'avg_incl_rate_pct': 5.49},
               'Qualitative': {'n_slr': 2, 'papers': 6536, 'avg_incl_rate_pct': 1.32}, 'Prognosis': {'n_slr': 1, 'papers': 3367, 'avg_incl_rate_pct': 5.70}}
LGAR_TABLE2_TAR2019 = {'MAP': 50.6, 'TNR@95%': 76.5, 'R@10%': 76.7, 'R@20%': 88.3, 'row': 'LGAR (T+R, monoT5)', 'source': 'ACL Anthology 2025.findings-acl.412 pp 7910-7927, Table 2 (macro-averages over 31 SLRs, footnote 8)'}
out = {'lgar_table1_as_published': LGAR_TABLE1, 'lgar_table2_tar2019_as_published': LGAR_TABLE2_TAR2019, 'qrels': {}}
for kind in ['abs', 'content']:
    out['qrels'][kind] = {}
    for typ in ['DTA', 'Intervention', 'Prognosis', 'Qualitative']:
        tot, pos = collections.Counter(), collections.Counter()
        for f in glob.glob(f'{ROOT}/clef/tar/2019-TAR/Task2/Testing/{typ}/qrels/*.{kind}.*'):
            for line in open(f):
                p = line.split()
                if len(p) < 4: continue
                tot[p[0]] += 1; pos[p[0]] += int(p[3])
        rates = [pos[t] / tot[t] for t in tot]
        out['qrels'][kind][typ] = {'n_slr': len(tot), 'papers': sum(tot.values()), 'positives': sum(pos.values()), 'macro_incl_rate_pct': round(100 * sum(rates) / len(rates), 2), 'pooled_incl_rate_pct': round(100 * sum(pos.values()) / sum(tot.values()), 2)}
# match
match = {}
for kind in ['abs', 'content']:
    match[kind] = all(abs(out['qrels'][kind][t]['macro_incl_rate_pct'] - LGAR_TABLE1[t]['avg_incl_rate_pct']) < 0.005 for t in LGAR_TABLE1)
out['table1_matches'] = match
out['conclusion'] = ('LGAR Table 1 average inclusion rates equal the macro-averaged inclusion rates of the abstract-level qrels (qrel_abs_test) to two decimals for all four review types and do not match the content-level qrels; '
                     'LGAR therefore evaluated on title-and-abstract screening labels, the same level as the manuscript\'s "title and abstract labels" row (31 reviews).')
# project files: positives at each level among retrievable records
R = json.load(open(f'{ROOT}/clef/clef_records.json'))
by = collections.defaultdict(lambda: [0, 0, 0])
for r in R:
    b = by[r['review']]; b[0] += 1; b[1] += int(r['label'] or 0); b[2] += int(r['label_ta'] or 0)
out['project_clef_records'] = {'topics': len(by), 'records': len(R), 'content_level_positives': sum(v[1] for v in by.values()), 'abstract_level_positives': sum(v[2] for v in by.values()),
                               'topics_without_content_level_positive': [k for k, v in by.items() if v[1] == 0]}
# existing computed comparison (read only)
C = json.load(open(f'{ROOT}/synergy/clef_results.json'))
out['manuscript_comparison_same_level'] = {'label': 'title and abstract labels (label_ta), 31 topics, macro-averages', 'jev': C['zeroshot_label_ta']['jev'], 'bge': C['zeroshot_label_ta']['bge'], 'lgar_reported': {k: v / 100 for k, v in LGAR_TABLE2_TAR2019.items() if isinstance(v, float)}, 'source': 'synergy/clef_results.json zeroshot_label_ta (unchanged, not recomputed)'}
out['manuscript_comparison_content_level'] = {'label': 'final inclusion (label), 28 topics', 'jev': C['zeroshot_label']['jev'], 'bge': C['zeroshot_label']['bge'], 'lgar_reported': None, 'source': 'synergy/clef_results.json zeroshot_label'}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lgar_label_level_check.json'), 'w'), indent=1)
print(json.dumps({'table1_matches': match, 'qrels_abs': {t: out['qrels']['abs'][t]['macro_incl_rate_pct'] for t in LGAR_TABLE1}, 'qrels_content': {t: out['qrels']['content'][t]['macro_incl_rate_pct'] for t in LGAR_TABLE1}}, indent=1))
