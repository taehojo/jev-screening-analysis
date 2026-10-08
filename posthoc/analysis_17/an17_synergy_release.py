# AN-0001-17: cite the SYNERGY release precisely and describe the split (documentary), compare development and held-out reviews.
import json, os, re, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
S = '/N/project/AiLab/jev/synergy/'; OUT = os.path.dirname(os.path.abspath(__file__))
rel = json.load(open(OUT + '/sources/github_api_releases_asreview_synergy-dataset.json'))
v3 = [r for r in rel if r['tag_name'] == 'synergy_plus_v3.0'][0]; v1 = [r for r in rel if r['tag_name'] == 'v1.0'][0]
tree = json.load(open(OUT + '/sources/github_api_tree_synergy_plus_v3.0.json')); paths = [x['path'] for x in tree['tree']]
top = sorted({p.split('/')[0] for p in paths}); pypi = json.load(open(OUT + '/sources/pypi_synergy-dataset.json'))
res = {'release': {'name': v3['name'], 'tag': v3['tag_name'], 'published_at_utc': v3['published_at'], 'url': v3['html_url'], 'body': v3['body'], 'repository': 'https://github.com/asreview/synergy-dataset (the former systematic-review-datasets repository redirects here)',
                   'tag_tree_dataset_folders': len(top), 'tag_tree_files_per_dataset': sorted({p.split('/', 1)[1] for p in paths if '/' in p and p.split('/')[0] == 'Abgaz_2023'}), 'top_level_documentation_files_in_tag': [p for p in paths if '/' not in p],
                   'v1_release': {'tag': v1['tag_name'], 'name': v1['name'], 'published_at_utc': v1['published_at'], 'url': v1['html_url']},
                   'fetched': '2026-09-26 17:52 to 17:54 EDT (GitHub API; copies in sources/)'},
       'package': {'name': 'synergy-dataset', 'version_used': '2.2', 'pypi_upload_2.2': pypi['releases']['2.2'][0]['upload_time'], 'pypi_upload_2.0.1': pypi['releases']['2.0.1'][0]['upload_time'], 'download_log': 'synergy/get_data.log: "Downloading version 3.0 of the SYNERGY+ dataset" (620 files)',
                   'default_version_in_package': 'base.py: "3.0" for synergy_plus', 'dataverse_doi_in_package': 'base.py: 10.34894/DDCVCV for SYNERGY+ (10.34894/HE6NAQ is the original SYNERGY v1 deposit cited as reference 18)'},
       'split_documentation': {'where_defined': 'synergy_dataset/splits.py: TEST_SPLIT list of 23 keys with the comment "Official train/test split for the SYNERGY+ dataset (114 SLRs). TEST_SPLIT contains the test set; all other datasets form the training set."',
                               'construction_documented': False, 'evidence': 'Neither the package README (PyPI, 2.2), the repository README (master), the release notes of synergy_plus_v3.0 ("Version 3.0 of the SYNERGY+ Dataset") nor the tag tree (dataset folders only, no top-level documentation) describes how the split was formed (random, temporal or by domain).',
                               'dataverse': 'The DataverseNL records (HE6NAQ, DDCVCV) could not be retrieved from this server on 2026-09-26 (bot-check page returned); copies of the responses are in sources/.'}}
# development versus held-out comparison
m = pd.read_csv(S + 'data/metadata/review_metadata.csv'); m['year'] = m.key.map(lambda k: int(re.search(r'(\d{4})[a-z]?$', k).group(1)))
m['prevalence'] = m.n_records_included / m.n_records; m['criteria_chars'] = m.eligibility_criteria.fillna('').astype(str).str.len()
dev_used = sorted({r['review'] for r in json.load(open(S + 'dev2000_records.json'))})
grp = {'held_out_23': m[m.split == 'test'], 'development_73_used': m[m.key.isin(dev_used)], 'development_all_91': m[m.split == 'train'], 'development_18_not_scored': m[(m.split == 'train') & (~m.key.isin(dev_used))]}
def desc(df):
    return {'n': len(df), 'records_median': float(df.n_records.median()), 'records_min': int(df.n_records.min()), 'records_max': int(df.n_records.max()), 'records_total': int(df.n_records.sum()), 'included_total': int(df.n_records_included.sum()),
            'prevalence_median_pct': round(float(100 * df.prevalence.median()), 2), 'prevalence_min_pct': round(float(100 * df.prevalence.min()), 2), 'prevalence_max_pct': round(float(100 * df.prevalence.max()), 2),
            'year_median': float(df.year.median()), 'year_min': int(df.year.min()), 'year_max': int(df.year.max()), 'year_counts': {str(k): int(v) for k, v in df.year.value_counts().sort_index().items()},
            'review_type_counts': {str(k): int(v) for k, v in df.review_type.value_counts().items()}, 'has_ti_ab_labels': int(df.has_ti_ab_labels.sum()), 'protocol_flag_1': int((df.protocol == 1).sum()),
            'criteria_chars_median': float(df.criteria_chars.median()), 'journal_counts_top5': {str(k): int(v) for k, v in df.journal.value_counts().head(5).items()}}
res['comparison'] = {k: desc(v) for k, v in grp.items()}
a, b = grp['held_out_23'], grp['development_73_used']
res['comparison']['tests_held_out_vs_development_73'] = {v: {'mannwhitney_p': round(float(mannwhitneyu(a[v], b[v], alternative='two-sided').pvalue), 4)} for v in ['n_records', 'prevalence', 'year', 'criteria_chars']}
a, b = grp['held_out_23'], grp['development_all_91']
res['comparison']['tests_held_out_vs_development_91'] = {v: {'mannwhitney_p': round(float(mannwhitneyu(a[v], b[v], alternative='two-sided').pvalue), 4)} for v in ['n_records', 'prevalence', 'year', 'criteria_chars']}
m[['key', 'split', 'n_records', 'n_records_included', 'prevalence', 'year', 'review_type', 'has_ti_ab_labels', 'protocol', 'criteria_chars', 'journal']].assign(used_in_development_73=m.key.isin(dev_used)).to_csv(OUT + '/review_level_comparison.csv', index=False)
json.dump(res, open(OUT + '/results.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
