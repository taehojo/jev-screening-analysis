# AN-0001-07 step 1: included studies below tau = 0.07 in the three CLEF reviews where the threshold rule fell below 95% recall.
# Reads original outputs read-only; fetches public PubMed summaries for the missed PMIDs (public identifiers only).
import json, os, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET, numpy as np
ROOT = '/N/project/AiLab/jev'; OUT = os.path.dirname(os.path.abspath(__file__)); TAU0 = 0.07; REVS = ['CD012233', 'CD011977', 'CD010558']
rec = [r for r in json.load(open(f'{ROOT}/clef/clef_records.json')) if r['review'] in REVS]
jev = {d['id']: d['p'] for d in json.load(open(f'{ROOT}/synergy/jev_clef.json')) if d.get('ok')}
crit = json.load(open(f'{ROOT}/clef/clef_criteria.json')); src = json.load(open(f'{ROOT}/clef/clef_criteria_source.json'))
out = {'analysis': 'AN-0001-07', 'tau': TAU0, 'reviews': {}}
for cd in REVS:
    rs = [r for r in rec if r['review'] == cd]; p = np.array([jev[r['id']] for r in rs]); y = np.array([int(r['label'] or 0) for r in rs]); order = np.argsort(-p, kind='stable'); rank = np.empty(len(rs), int); rank[order] = np.arange(1, len(rs) + 1)
    inc = [i for i in range(len(rs)) if y[i] == 1]; s = p >= TAU0
    info = {'type': rs[0]['split'], 'N': len(rs), 'n_included': int(y.sum()), 'n_ta_positive': int(sum(int(r['label_ta'] or 0) for r in rs)), 'n_read_at_tau': int(s.sum()), 'work_at_tau': float(s.mean()), 'recall_at_tau': float(y[s].sum() / y.sum()), 'n_included_below_tau': int(((~s) & (y == 1)).sum()),
            'included_p_sorted': sorted(float(p[i]) for i in inc), 'criteria_source_pmid': src[cd]['pmid'], 'criteria_source_year': src[cd]['year'], 'criteria_text': crit[cd],
            'largest_tau_keeping_95_recall': float(sorted(p[inc])[int(np.floor(0.05 * len(inc) + 1e-9))]), 'n_ta_positive_below_tau': int(sum(1 for i in range(len(rs)) if int(rs[i]['label_ta'] or 0) == 1 and p[i] < TAU0)),
            'missed': [{'pmid': rs[i]['id'].split('|')[1], 'p': float(p[i]), 'rank': int(rank[i]), 'rank_pct': float(rank[i] / len(rs)), 'label_ta': rs[i]['label_ta'], 'title': rs[i]['title'], 'abstract': rs[i]['abstract']} for i in inc if p[i] < TAU0]}
    out['reviews'][cd] = info
# PubMed summaries for the missed studies (public PMIDs)
pm = [m['pmid'] for cd in REVS for m in out['reviews'][cd]['missed']]
base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
import subprocess
url = base + 'efetch.fcgi?' + urllib.parse.urlencode({'db': 'pubmed', 'id': ','.join(pm), 'retmode': 'xml'})
raw = subprocess.run(['curl', '-s', '-m', '120', url], capture_output=True, check=True).stdout   # curl used because python urllib fails TLS verification on this server
open(f'{OUT}/sources/missed_pubmed_efetch.xml', 'wb').write(raw); x = ET.fromstring(raw)
meta = {}
for art in x.findall('.//PubmedArticle'):
    pid = art.findtext('.//MedlineCitation/PMID'); au = art.findall('.//AuthorList/Author'); first = (au[0].findtext('LastName') or au[0].findtext('CollectiveName') or '') if au else ''
    meta[pid] = {'first_author': first, 'year': art.findtext('.//PubDate/Year') or art.findtext('.//PubDate/MedlineDate') or '', 'journal': art.findtext('.//Journal/ISOAbbreviation') or '', 'doi': next((i.text for i in art.findall('.//ArticleIdList/ArticleId') if i.get('IdType') == 'doi'), None), 'pub_types': [t.text for t in art.findall('.//PublicationType')]}
for cd in REVS:
    for m in out['reviews'][cd]['missed']: m.update(meta.get(m['pmid'], {}))
out['pubmed_fetch_time'] = time.strftime('%Y-%m-%d %H:%M:%S %Z')
json.dump(out, open(f'{OUT}/missed_studies.json', 'w'), indent=1, ensure_ascii=False)
for cd in REVS:
    v = out['reviews'][cd]; print(cd, v['type'], 'N', v['N'], 'included', v['n_included'], 'below tau', v['n_included_below_tau'], 'recall', round(v['recall_at_tau'], 4), 'work', round(v['work_at_tau'], 4), 'max safe tau', v['largest_tau_keeping_95_recall'], 'TA pos', v['n_ta_positive'], 'TA pos below tau', v['n_ta_positive_below_tau'])
    print('  included p:', v['included_p_sorted'])
    for m in v['missed']: print('  MISSED', m['pmid'], m['p'], 'rank', m['rank'], f"({m['rank_pct']:.3f})", 'ta', m['label_ta'], m.get('first_author'), m.get('year'), m.get('journal'), '|', m['title'][:110])
