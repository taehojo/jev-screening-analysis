#!/usr/bin/env python3
# npj AN-0001-15 stage A (feasibility, no model call): retrieve the protocol version of each CLEF 2019 review with at least one
# content-relevant record (28). Protocol DOI = 10.1002/14651858.CDxxxxxx (no .pub suffix).
# Sources tried per review, in this order, each once (1 s between requests): Europe PMC REST search on the protocol DOI;
# PubMed E-utilities search (CD number in the Cochrane journal) with efetch of the hits to find a record whose DOI has no .pub suffix;
# Europe PMC full text XML and the PMC article page if a PMCID exists; Cochrane Library and Wiley pages of the protocol DOI.
# Every response is saved under sources/ with its HTTP status in sources/fetch_record.tsv. Only public identifiers are sent.
import os, re, json, time, ssl, subprocess, urllib.request, urllib.parse, urllib.error
CTX = ssl.create_default_context(cafile='/etc/pki/tls/certs/ca-bundle.crt')   # system CA bundle (the conda Python has none)
import xml.etree.ElementTree as ET
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(HERE, 'sources'); os.makedirs(SRC, exist_ok=True)
ROOT = '/N/project/AiLab/jev'
UA = 'Mozilla/5.0 (research script; systematic review screening benchmark; contact via repository)'
def now(): return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S %Z'], capture_output=True, text=True).stdout.strip()
REC = open(os.path.join(SRC, 'fetch_record.tsv'), 'a')
def get(url, fname, accept=None):
    time.sleep(1.0); h = {'User-Agent': UA}
    if accept: h['Accept'] = accept
    t = now()
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60, context=CTX); body = r.read(); st = r.status
    except urllib.error.HTTPError as e:
        body = e.read() if hasattr(e, 'read') else b''; st = e.code
    except Exception as e:
        body = repr(e).encode(); st = -1
    open(os.path.join(SRC, fname), 'wb').write(body)
    REC.write(f'{t}\t{st}\t{len(body)}\t{fname}\t{url}\n'); REC.flush()
    return st, body

by = {}
for r in json.load(open(ROOT + '/clef/clef_records.json')): by.setdefault(r['review'], []).append(r)
revs = sorted(k for k, v in by.items() if any(r['label'] for r in v))
out = {'started': now(), 'reviews': {}}
EU = 'https://www.ebi.ac.uk/europepmc/webservices/rest/'
EB = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
for cd in revs:
    doi = f'10.1002/14651858.{cd}'; R = {'protocol_doi': doi, 'n_records': len(by[cd])}
    # 1. Europe PMC search on the protocol DOI (resultType lite; one retry after 10 s on HTTP 503)
    url = EU + 'search?' + urllib.parse.urlencode({'query': f'DOI:"{doi}"', 'format': 'json', 'resultType': 'lite'}, safe='/:')
    st, b = get(url, f'{cd}_europepmc_search.json')
    if st == 503: time.sleep(10); st, b = get(url, f'{cd}_europepmc_search_retry.json')
    R['europepmc_status'] = st
    hits = []
    try: hits = json.loads(b)['resultList']['result']
    except Exception: pass
    R['europepmc_hits'] = [{'id': h.get('id'), 'source': h.get('source'), 'pmid': h.get('pmid'), 'pmcid': h.get('pmcid'), 'doi': h.get('doi'), 'title': h.get('title'),
                            'pubYear': h.get('pubYear'), 'inPMC': h.get('inPMC'), 'isOpenAccess': h.get('isOpenAccess')} for h in hits]
    # 2. PubMed: all CDSR records of this CD number, each with its own DOI and PMCID (PubmedData/ArticleIdList only, not the reference list)
    st, b = get(EB + 'esearch.fcgi?' + urllib.parse.urlencode({'db': 'pubmed', 'term': f'{cd}[All Fields] AND "Cochrane Database Syst Rev"[Journal]', 'retmax': 50, 'retmode': 'json'}), f'{cd}_pubmed_esearch.json')
    ids = []
    try: ids = json.loads(b)['esearchresult']['idlist']
    except Exception: pass
    R['pubmed_ids'] = ids; R['pubmed_versions'] = []; R['pubmed_protocol'] = None
    if ids:
        st, b = get(EB + 'efetch.fcgi?' + urllib.parse.urlencode({'db': 'pubmed', 'id': ','.join(ids), 'retmode': 'xml'}), f'{cd}_pubmed_efetch.xml')
        try:
            for art in ET.fromstring(b).findall('PubmedArticle'):
                own = art.find('PubmedData/ArticleIdList')
                idl = {e.get('IdType'): e.text for e in (own if own is not None else [])}
                eloc = [e.text for e in art.findall('MedlineCitation/Article/ELocationID') if e.get('EIdType') == 'doi']
                d_ = (idl.get('doi') or (eloc[0] if eloc else '') or '').lower()
                ab = ' '.join(''.join(a.itertext()) for a in art.findall('MedlineCitation/Article/Abstract/AbstractText'))
                ti = art.findtext('MedlineCitation/Article/ArticleTitle') or ''
                v = {'pmid': art.findtext('MedlineCitation/PMID'), 'doi': d_, 'pmc': idl.get('pmc'), 'title': ti, 'year': art.findtext('MedlineCitation/Article/Journal/JournalIssue/PubDate/Year'),
                     'is_protocol': ('this is a protocol' in ab.lower()), 'abstract_start': ab[:300]}
                R['pubmed_versions'].append(v)
                if v['is_protocol'] and d_ == doi.lower(): R['pubmed_protocol'] = v
        except Exception as e: R['pubmed_parse_error'] = repr(e)
    # 3. full text if a PMCID exists
    pp = R['pubmed_protocol'] or {}
    pmcids = sorted({h['pmcid'] for h in R['europepmc_hits'] if h.get('pmcid') and (h.get('doi') or '').lower() == doi.lower() and pp and h.get('pmid') == pp.get('pmid')} | ({pp['pmc']} if pp.get('pmc') else set()))
    R['pmcids'] = pmcids; R['fulltext'] = []
    for pmc in pmcids:
        st, b = get(EU + f'{pmc}/fullTextXML', f'{cd}_{pmc}_europepmc_fullTextXML.xml'); R['fulltext'].append({'pmcid': pmc, 'source': 'europepmc_fullTextXML', 'status': st, 'bytes': len(b)})
        st, b = get(f'https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/', f'{cd}_{pmc}_pmc.html'); R['fulltext'].append({'pmcid': pmc, 'source': 'pmc_html', 'status': st, 'bytes': len(b)})
    # 4. publisher pages (recorded; earlier runs from this server returned 412 and 403)
    st, b = get(f'https://www.cochranelibrary.com/cdsr/doi/{doi}/full', f'{cd}_cochranelibrary.html'); R['cochrane_library_status'] = st
    st, b = get(f'https://onlinelibrary.wiley.com/doi/full/{doi}', f'{cd}_wiley.html'); R['wiley_status'] = st
    out['reviews'][cd] = R
    R['protocol_identified'] = bool(R['pubmed_protocol'])
    print(cd, 'protocol identified', R['protocol_identified'], 'EPMC hits', len(R['europepmc_hits']), 'PubMed versions', [(v['doi'][-8:], v['year'], v['is_protocol'], v['pmc']) for v in R['pubmed_versions']], 'PMCIDs', pmcids, 'CL', R['cochrane_library_status'], 'Wiley', R['wiley_status'], flush=True)
out['finished'] = now()
json.dump(out, open(os.path.join(HERE, 'stageA_retrieval.json'), 'w'), indent=1)
