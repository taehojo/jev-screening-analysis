# AN-0001-18: build the reference verification table from the API responses saved in api_responses/ (no network access needed to re-run).
# Sources: Crossref works API, PubMed E-utilities efetch, arXiv API, medRxiv (api.medrxiv.org), DataCite, CEUR-WS index, TypeSafe blog page (all fetched 2026-09-26 by curl; see EXECUTION_RECORD.md).
import json, glob, os, re, csv, xml.etree.ElementTree as ET
H = os.path.dirname(os.path.abspath(__file__)); A = os.path.join(H, 'api_responses')
cr = {}
for f in glob.glob(A + '/crossref_*.json'):
    try:
        m = json.load(open(f))['message']
        if 'DOI' in m: cr[m['DOI'].lower()] = m
    except Exception: pass
def crf(doi):
    m = cr.get(doi.lower()); 
    if not m: return None
    dp = lambda k: '-'.join(str(x) for x in m.get(k, {}).get('date-parts', [[None]])[0]) if m.get(k, {}).get('date-parts', [[None]])[0][0] else None
    return {'type': m.get('type'), 'title': (m.get('title') or [''])[0], 'container': (m.get('container-title') or [''])[0], 'volume': m.get('volume'), 'issue': m.get('issue'), 'pages': m.get('page'), 'article_number': m.get('article-number'), 'published': dp('published'), 'online': dp('published-online'), 'authors': [(a.get('family', '') + ' ' + ''.join(w[0] for w in a.get('given', '').replace('-', ' ').split())).strip() for a in m.get('author', [])]}
pm = {}
for f in glob.glob(A + '/pubmed_efetch_*.xml'):
    try: root = ET.parse(f).getroot()
    except Exception: continue
    for art in root.findall('.//PubmedArticle'):
        j = art.find('.//Journal'); ad = art.find('.//ArticleDate')
        pm[art.findtext('.//PMID')] = {'title': art.findtext('.//ArticleTitle'), 'journal': j.findtext('.//ISOAbbreviation'), 'year': j.findtext('.//JournalIssue/PubDate/Year'), 'volume': j.findtext('.//JournalIssue/Volume'), 'issue': j.findtext('.//JournalIssue/Issue'), 'pages': art.findtext('.//MedlinePgn'), 'doi': next((e.text for e in art.findall('.//ArticleIdList/ArticleId') if e.get('IdType') == 'doi'), None), 'status': art.findtext('.//PublicationStatus'), 'epub': '-'.join(ad.findtext(t) for t in ('Year', 'Month', 'Day')) if ad is not None else None}
ns = {'a': 'http://www.w3.org/2005/Atom'}; ax = {}
for f in glob.glob(A + '/arxiv_*.xml'):
    for e in ET.parse(f).getroot().findall('a:entry', ns):
        ax[re.sub(r'v\d+$', '', e.findtext('a:id', namespaces=ns).split('/abs/')[-1])] = {'title': ' '.join(e.findtext('a:title', namespaces=ns).split()), 'published': e.findtext('a:published', namespaces=ns)}
mrx = json.load(open(A + '/medrxiv_api2.json'))['collection']
dc = json.load(open(A + '/datacite_10.34894_HE6NAQ.json'))['data']['attributes']
refs = [
 (1, 'Borah 2017 BMJ Open 7: e012545', '10.1136/bmjopen-2016-012545', '28242767'), (2, 'Cohen 2006 JAMIA 13: 206-19', '10.1197/jamia.M1929', '16357352'), (3, 'Harasgama 2026 JMIR 28: e81597', '10.2196/81597', '41911537'),
 (4, 'van de Schoot 2021 Nat Mach Intell 3: 125-33', '10.1038/s42256-020-00287-7', None), (5, 'Ferdinands 2023 Syst Rev 12: 100', '10.1186/s13643-023-02257-7', '37340494'), (6, 'Cormack 2016 SIGIR 75-84', '10.1145/2911451.2911510', None),
 (7, 'Callaghan 2020 Syst Rev 9: 273', '10.1186/s13643-020-01521-4', '33248464'), (8, 'Repke 2026 Cochrane Evid Synth Methods 4: e70068', '10.1002/cesm.70068', '41583534'), (9, 'Wang 2024 ECIR LNCS 403-20', '10.1007/978-3-031-56027-9_25', None),
 (10, 'Jaumann 2025 Findings ACL (no pages in v0)', '10.18653/v1/2025.findings-acl.412', None), (11, 'Akinseloyin 2024 JAMIA 31: 1939-52', '10.1093/jamia/ocae166', '39042516'), (12, 'Akinseloyin 2025 medRxiv (title as in v0: LLM-guided weak supervision ...; published online Aug 26)', '10.1101/2025.08.24.25334314', None),
 (13, 'Li 2026 JAMIA 33: 1026-36', '10.1093/jamia/ocag014', '41801982'), (14, 'Pitre 2026 J Clin Epidemiol 112514', '10.1016/j.jclinepi.2026.112514', '42767575'), (15, 'Janoudi 2025 Value Health 28: 1630-36', '10.1016/j.jval.2025.09.008', '40998117'),
 (16, 'TypeSafe AI blog, Sept 15, 2026', None, None), (17, 'Gallifant 2025 Nat Med 31: 60-69', '10.1038/s41591-024-03425-5', '39779929'), (18, 'SYNERGY DataverseNL 2023', '10.34894/HE6NAQ', None), (19, 'Kanoulas 2019 CEUR 2380', None, None),
 (20, 'Hida 2026 arXiv 2608.14737', None, None), (21, 'Xiao 2023 arXiv 2309.07597', None, None), (22, 'Kusa 2023 ICTIR (no pages in v0)', '10.1145/3578337.3605135', None),
 ('C1', 'candidate: Van Calster 2019 BMC Med 17: 230', '10.1186/s12916-019-1466-7', '31842878'), ('C2', 'candidate: Yang, Lewis, Frieder 2021 DocEng 1-10', '10.1145/3469096.3469873', None)]
rows = []
for n, cite, doi, pmid in refs:
    r = {'ref': n, 'v0_citation_summary': cite, 'doi': doi, 'pmid': pmid, 'crossref': json.dumps(crf(doi), ensure_ascii=False) if doi else None, 'pubmed': json.dumps(pm.get(pmid), ensure_ascii=False) if pmid else None, 'other_source': None, 'status': 'VERIFIED', 'note': ''}
    if n == 10: r['note'] = 'pages 7910-7927 (Crossref; ACL Anthology BibTeX) to be added'
    if n == 22: r['note'] = 'pages 125-133 (Crossref) to be added'
    if n == 12: r['other_source'] = json.dumps([{k: c[k] for k in ('title', 'date', 'version')} for c in mrx]); r['status'] = 'EXISTS_CITATION_MISMATCH'; r['note'] = 'v0 pairs the version 2 title (posted 2026-05-10; also the Crossref title) with the version 1 date (2025-08-26); version 1 is titled "Weakly Supervised Active Learning for Abstract Screening Leveraging LLM-Based Pseudo-Labeling"'
    if n == 14: r['note'] = 'journal article confirmed: Crossref journal-article, PubMed 42767575 ahead of print, electronic publication 2026-09-21 (accepted 2026-09-09); SSRN preprint 10.2139/ssrn.6732292 also exists; abstract states median AUC 0.974 across 3,434 test reviews and 0.976 on 22 external reviews'
    if n == 16: r['other_source'] = 'typesafe_blog.html fetched 2026-09-26: <title>Introducing System One Models & Jev - TypeSafe AI Blog</title>; page shows "Sep 15, 2026"'
    if n == 18: r['other_source'] = json.dumps({'titles': [t.get('title') for t in dc.get('titles', [])], 'publisher': dc.get('publisher'), 'year': dc.get('publicationYear'), 'creators': [c.get('name') for c in dc.get('creators', [])], 'version': dc.get('version')})
    if n == 19: r['other_source'] = 'CEUR-WS Vol-2380 index (ceur_vol2380_index.html in AN-0001-19/sources): paper_250 "CLEF 2019 Technology Assisted Reviews in Empirical Medicine Overview", Kanoulas, Li, Azzopardi, Spijker; Working Notes of CLEF 2019, Lugano, Sept 9-12, 2019; volume published 2019-07-23'; r['note'] = 'URL https://ceur-ws.org/Vol-2380/paper_250.pdf may be added'
    if n == 20: r['other_source'] = json.dumps(ax.get('2608.14737'))
    if n == 21: r['other_source'] = json.dumps(ax.get('2309.07597'))
    if n == 'C1': r['note'] = 'abstract (PubMed) supports citing it for calibration assessment at validation (calibration curves, updating); add only if cited in the revision'
    if n == 'C2': r['other_source'] = 'arXiv 2106.09871 abstract: proposes and compares heuristic stopping rules for TAR against recall targets'; r['note'] = 'Crossref record has no abstract; arXiv abstract supports citing it for heuristic stopping rules (Quant, QuantCI, comparison of heuristics); a statement about the knee method specifically must be checked against the full text before citing'
    rows.append(r)
with open(os.path.join(H, 'references_verified.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('rows', len(rows)); print({r['ref']: r['status'] for r in rows})
