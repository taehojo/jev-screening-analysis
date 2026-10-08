# SYNERGY 3.0 -> records.json (id, review, split, title, abstract, label) + criteria.json (리뷰별 고정 프롬프트 문단)
import pandas as pd, json, glob, os, re
m = pd.read_csv('data/metadata/review_metadata.csv')
recs, crit, summ = [], {}, []
def clean(x):
    return '' if pd.isna(x) else re.sub(r'\s+', ' ', str(x)).strip()
for _, r in m.iterrows():
    k = r['key']
    d = pd.read_csv(f'data/{k}.csv')
    crit[k] = (f"Systematic review title: {clean(r['title'])}\n"
               f"Research question: {clean(r['research_question']) or '(not stated)'}\n"
               f"Eligibility criteria reported by the review authors:\n{str(r['eligibility_criteria']).strip()}")
    n_noabs = 0
    for i, x in d.iterrows():
        a = clean(x.get('abstract')); t = clean(x.get('title'))
        if not a: n_noabs += 1
        rid = clean(x.get('openalex_id')) or f'{k}#{i}'
        recs.append({'id': f'{k}|{rid}|{i}', 'review': k, 'split': r['split'], 'title': t, 'abstract': a,
                     'label': int(x['label_included']),
                     'label_ta': (int(x['label_abstract_included']) if 'label_abstract_included' in d.columns and not pd.isna(x.get('label_abstract_included')) else None)})
    summ.append({'review': k, 'split': r['split'], 'n': len(d), 'incl': int(d['label_included'].sum()), 'no_abstract': n_noabs,
                 'has_ta': int('label_abstract_included' in d.columns), 'crit_chars': len(crit[k])})
json.dump(recs, open('records.json', 'w'))
json.dump(crit, open('criteria.json', 'w'), indent=1)
S = pd.DataFrame(summ); S.to_csv('review_summary.csv', index=False)
print('records', len(recs), 'reviews', len(crit), 'included', sum(r['label'] for r in recs))
print('no-abstract records', S.no_abstract.sum(), f'({100*S.no_abstract.sum()/len(recs):.1f}%)')
print(S.groupby('split').agg(n=('n','sum'), incl=('incl','sum'), reviews=('review','count'), noabs=('no_abstract','sum')).to_string())
print('ta-label columns present in', S.has_ta.sum(), 'reviews')
import statistics
w = [len((r['title'] + ' ' + r['abstract']).split()) for r in recs]; print('title+abstract words median', statistics.median(w), 'p95', sorted(w)[int(.95*len(w))])
