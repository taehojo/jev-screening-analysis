# CLEF eHealth TAR 2019 Task2 평가 세트 → records.json, criteria.json (전부 공개 자료)
# 적격 기준: 코크란 리뷰 PubMed 초록의 OBJECTIVES, SELECTION CRITERIA 절만 사용 (결과 절은 포함 연구 정보가 있어 제외)
import os, re, json, time, glob, urllib.request, urllib.parse, xml.etree.ElementTree as ET
B = 'tar/2019-TAR/Task2/Testing'
base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
def get(url, data=None):
    for a in range(6):
        try:
            req = urllib.request.Request(url, data=data.encode() if data else None)
            return urllib.request.urlopen(req, timeout=120).read()
        except Exception as e:
            err = e; time.sleep(2 * (a + 1))
    raise RuntimeError(err)
topics = {}
for typ in ['DTA', 'Intervention', 'Prognosis', 'Qualitative']:
    for f in glob.glob(f'{B}/{typ}/topics/*'):
        t = open(f).read(); cd = re.search(r'Topic:\s*(CD\d+)', t).group(1); title = re.search(r'Title:\s*(.*)', t).group(1).strip()
        topics[cd] = {'type': typ, 'title': title}
    for kind in ['abs', 'content']:
        for line in open(glob.glob(f'{B}/{typ}/qrels/*.{kind}.*')[0]):
            p = line.split()
            if len(p) < 4: continue
            topics[p[0]].setdefault(kind, {})[p[2]] = int(p[3])
print('topics', len(topics), {k: len(v['content']) for k, v in list(topics.items())[:3]})
# 1) 코크란 리뷰 초록에서 목적과 선정 기준
crit = {}
for cd, v in topics.items():
    q = f'{cd}[All Fields] AND "Cochrane Database Syst Rev"[Journal]'
    r = json.loads(get(base + 'esearch.fcgi', urllib.parse.urlencode({'db': 'pubmed', 'term': q, 'retmax': 50, 'retmode': 'json'})))
    ids = r['esearchresult']['idlist']; time.sleep(0.4)
    best = None
    if ids:
        x = ET.fromstring(get(base + 'efetch.fcgi', urllib.parse.urlencode({'db': 'pubmed', 'id': ','.join(ids), 'retmode': 'xml'}))); time.sleep(0.4)
        cands = []
        for art in x.findall('.//PubmedArticle'):
            secs = {(a.get('Label') or '').upper(): ''.join(a.itertext()).strip() for a in art.findall('.//Abstract/AbstractText')}
            yr = int((art.findtext('.//PubDate/Year') or art.findtext('.//ArticleDate/Year') or '0') or 0)
            if 'SELECTION CRITERIA' in secs and 'OBJECTIVES' in secs:
                cands.append((yr, art.findtext('.//PMID'), secs))
        pre = [c for c in cands if c[0] <= 2019]
        if cands: best = max(pre, key=lambda c: c[0]) if pre else min(cands, key=lambda c: c[0])
    if best:
        crit[cd] = {'pmid': best[1], 'year': best[0], 'text': f"Systematic review title: {v['title']}\nResearch question: {best[2]['OBJECTIVES']}\nEligibility criteria reported by the review authors:\n{best[2]['SELECTION CRITERIA']}"}
    print(cd, v['type'], 'criteria from PMID', best[1] if best else None, best[0] if best else '', flush=True)
json.dump({k: v['text'] for k, v in crit.items()}, open('clef_criteria.json', 'w'), indent=1)
json.dump({k: {'pmid': v['pmid'], 'year': v['year']} for k, v in crit.items()}, open('clef_criteria_source.json', 'w'), indent=1)
# 2) 모든 PMID의 제목·초록
allp = sorted({p for v in topics.values() for p in v['content']}); recs = {}
for i in range(0, len(allp), 200):
    x = ET.fromstring(get(base + 'efetch.fcgi', urllib.parse.urlencode({'db': 'pubmed', 'id': ','.join(allp[i:i + 200]), 'retmode': 'xml'}))); time.sleep(0.35)
    for art in x.findall('.//PubmedArticle'):
        pm = art.findtext('.//MedlineCitation/PMID'); t = art.find('.//ArticleTitle')
        recs[pm] = {'title': ' '.join(''.join(t.itertext()).split()) if t is not None else '', 'abstract': ' '.join(' '.join(''.join(a.itertext()) for a in art.findall('.//Abstract/AbstractText')).split())}
    if (i // 200) % 25 == 0: print(f'  fetched {len(recs)}/{len(allp)}', flush=True)
out = []
for cd, v in topics.items():
    if cd not in crit: continue
    for pm, lab in v['content'].items():
        r = recs.get(pm)
        if not r or not (r['title'] or r['abstract']): continue
        out.append({'id': f'{cd}|{pm}', 'review': cd, 'split': v['type'], 'title': r['title'], 'abstract': r['abstract'], 'label': lab, 'label_ta': v['abs'].get(pm)})
json.dump(out, open('clef_records.json', 'w'))
print('records', len(out), 'topics', len({r['review'] for r in out}), 'pos', sum(r['label'] for r in out), 'no-abstract', sum(1 for r in out if not r['abstract']), 'missing pmids', len(allp) - len(recs))
