# AN-0001-11: provenance and length of the eligibility criteria supplied to the models.
import json, os, re, numpy as np, pandas as pd
S = '/N/project/AiLab/jev/synergy/'; OUT = os.path.dirname(os.path.abspath(__file__))
m = pd.read_csv(S + 'data/metadata/review_metadata.csv')
crit = json.load(open(S + 'criteria.json'))
dev_used = sorted({r['review'] for r in json.load(open(S + 'dev2000_records.json'))})
rows = []
for _, r in m.iterrows():
    ec = str(r['eligibility_criteria']) if not pd.isna(r['eligibility_criteria']) else ''
    blk = crit[r['key']]
    rows.append({'review': r['key'], 'split': r['split'], 'used_in_development_73': r['key'] in dev_used, 'protocol_flag': r['protocol'], 'criteria_question_flag': r['criteria_question'], 'labels_match_flag': r['labels_match'],
                 'criteria_chars': len(ec), 'criteria_words': len(ec.split()), 'block_chars': len(blk), 'block_words': len(blk.split()), 'research_question_present': not pd.isna(r['research_question']) and str(r['research_question']).strip() != ''})
L = pd.DataFrame(rows); L.to_csv(OUT + '/per_review_criteria_length.csv', index=False)
def summ(df):
    return {'n': len(df), 'criteria_chars_median': float(df.criteria_chars.median()), 'criteria_chars_min': int(df.criteria_chars.min()), 'criteria_chars_max': int(df.criteria_chars.max()), 'criteria_chars_iqr': [float(df.criteria_chars.quantile(0.25)), float(df.criteria_chars.quantile(0.75))],
            'criteria_words_median': float(df.criteria_words.median()), 'criteria_words_min': int(df.criteria_words.min()), 'criteria_words_max': int(df.criteria_words.max()),
            'block_chars_median': float(df.block_chars.median()), 'block_chars_min': int(df.block_chars.min()), 'block_chars_max': int(df.block_chars.max()), 'block_words_median': float(df.block_words.median()),
            'protocol_flag_1': int((df.protocol_flag == 1).sum()), 'protocol_flag_0': int((df.protocol_flag == 0).sum()), 'research_question_present': int(df.research_question_present.sum())}
res = {'synergy': {'held_out_23': summ(L[L.split == 'test']), 'development_73_used': summ(L[L.used_in_development_73]), 'development_all_91': summ(L[L.split == 'train']), 'all_114': summ(L)}}
res['synergy']['shortest_criteria_reviews'] = L.nsmallest(5, 'criteria_chars')[['review', 'split', 'criteria_chars', 'criteria_words']].to_dict(orient='records')
res['synergy']['longest_criteria_reviews'] = L.nlargest(5, 'criteria_chars')[['review', 'split', 'criteria_chars', 'criteria_words']].to_dict(orient='records')
# provenance evidence
res['synergy_provenance'] = {
    'deposit_metadata_json': 'In the SYNERGY+ 3.0 deposit downloaded on 2026-09-24 (~/.synergy_dataset_source/synergy-dataset-plus/<key>/metadata.json), eligibility_criteria sits under the "publication" object together with the DOI of the review publication; example key Abgaz_2023.',
    'package_readme': 'synergy-dataset 2.2 README (PyPI, fetched 2026-09-26): "eligibility_criteria: the screening criteria text from metadata.json (overwritten by reviews.csv Eligibility Criteria, if available)". No statement that criteria were taken from protocols.',
    'classic_datasets_toml': 'The repository file datasets.toml (master branch, fetched 2026-09-26) stores eligibility_criteria under [datasets.publication].',
    'reviews_csv_protocol_column': 'reviews.csv has a column "Protocol" (0/1) with no definition in the files available to us; its distribution is reported but not interpreted as the source of the criteria text.',
    'conclusion': 'The metadata attributes the criteria text to the review publication; whether the text was drafted at protocol stage cannot be determined from the dataset documentation.',
    'example_text_evidence': 'Abgaz_2023 criteria text refers to "details of which are provided at the end of this subsection", wording copied from the published paper.'}
# CLEF
cc = json.load(open('/N/project/AiLab/jev/clef/clef_criteria.json')); cs = json.load(open('/N/project/AiLab/jev/clef/clef_criteria_source.json'))
rc = []
for k, t in cc.items():
    parts = t.split('Eligibility criteria reported by the review authors:\n'); ec = parts[1] if len(parts) > 1 else ''
    rc.append({'topic': k, 'year_of_abstract_version': cs[k]['year'], 'pmid': cs[k]['pmid'], 'criteria_chars': len(ec), 'criteria_words': len(ec.split()), 'block_chars': len(t), 'block_words': len(t.split())})
C = pd.DataFrame(rc); C.to_csv(OUT + '/clef_criteria_length.csv', index=False)
res['clef'] = {'n': len(C), 'criteria_chars_median': float(C.criteria_chars.median()), 'criteria_chars_min': int(C.criteria_chars.min()), 'criteria_chars_max': int(C.criteria_chars.max()), 'criteria_words_median': float(C.criteria_words.median()), 'criteria_words_min': int(C.criteria_words.min()), 'criteria_words_max': int(C.criteria_words.max()),
               'block_chars_median': float(C.block_chars.median()), 'block_chars_min': int(C.block_chars.min()), 'block_chars_max': int(C.block_chars.max()),
               'abstract_version_years': {str(k): int(v) for k, v in C.year_of_abstract_version.value_counts().sort_index().items()},
               'source_rule': 'clef/build_clef.py: PubMed search "<CD number> AND Cochrane Database Syst Rev"; the Objectives and Selection criteria sections of the abstract of the latest version published by 2019, else the earliest available version; the Cochrane review abstracts describe the completed review, not the protocol.'}
# pilot review criteria (protocol-stage summary written by the authors)
pc = json.load(open(S + 'dhl_criteria.json'))['dhl']
res['pilot'] = {'block_chars': len(pc), 'block_words': len(pc.split()), 'source': 'Authors summary of the review protocol eligibility criteria (synergy/dhl_criteria.json; screen/run_screen.mjs CRITERIA)'}
json.dump(res, open(OUT + '/results.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
