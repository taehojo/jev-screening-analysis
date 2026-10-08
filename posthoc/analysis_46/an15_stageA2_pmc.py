#!/usr/bin/env python
# npj AN-0001-15 stage A, supplementary step (logged before the run): PMC holds Cochrane protocols as PMC-only records (no PMID).
# For each review: PMC esearch (CD number in the Cochrane journal), esummary of the hits (DOI, title, date, article ids);
# protocol = a PubMed record with the unsuffixed DOI whose abstract says "This is a protocol" or "This is the protocol"
# (stageA_retrieval.json), or a PMC record whose DOI is the unsuffixed DOI. For identified protocols the full text is fetched
# from PMC (article page) and Europe PMC (fullTextXML). Europe PMC searches of stage A that failed (HTTP 5xx) are repeated once.
import os, re, json, time, ssl, subprocess, urllib.request, urllib.parse, urllib.error
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(HERE, 'sources')
CTX = ssl.create_default_context(cafile='/etc/pki/tls/certs/ca-bundle.crt')
UA = 'Mozilla/5.0 (research script; systematic review screening benchmark)'
def now(): return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S %Z'], capture_output=True, text=True).stdout.strip()
REC = open(os.path.join(SRC, 'fetch_record.tsv'), 'a')
def get(url, fname):
    time.sleep(1.0); t = now()
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}), timeout=60, context=CTX); body = r.read(); st = r.status
    except urllib.error.HTTPError as e: body = e.read() if hasattr(e, 'read') else b''; st = e.code
    except Exception as e: body = repr(e).encode(); st = -1
    open(os.path.join(SRC, fname), 'wb').write(body); REC.write(f'{t}\t{st}\t{len(body)}\t{fname}\t{url}\n'); REC.flush()
    return st, body
EB = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'; EU = 'https://www.ebi.ac.uk/europepmc/webservices/rest/'
A = json.load(open(os.path.join(HERE, 'stageA_retrieval.json')))
PROT = re.compile(r'this is (a|the) protocol', re.I)
out = {'started': now(), 'reviews': {}}
for cd, R in A['reviews'].items():
    doi = R['protocol_doi'].lower(); S = {}
    pm = [v for v in R['pubmed_versions'] if v['doi'] == doi and PROT.search(v['abstract_start'] or '')]
    S['pubmed_protocol'] = pm[0] if pm else None
    if R.get('europepmc_status') != 200:
        st, b = get(EU + 'search?' + urllib.parse.urlencode({'query': f'DOI:"{R["protocol_doi"]}"', 'format': 'json', 'resultType': 'lite'}, safe='/:'), f'{cd}_europepmc_search_A2.json')
        S['europepmc_repeat_status'] = st
        try: S['europepmc_repeat_hits'] = [{k: h.get(k) for k in ['id', 'source', 'pmid', 'pmcid', 'doi', 'title', 'pubYear']} for h in json.loads(b)['resultList']['result']]
        except Exception: S['europepmc_repeat_hits'] = None
    st, b = get(EB + 'esearch.fcgi?' + urllib.parse.urlencode({'db': 'pmc', 'term': f'{cd}[All Fields] AND "Cochrane Database Syst Rev"[Journal]', 'retmax': 50, 'retmode': 'json'}), f'{cd}_pmc_esearch.json')
    ids = []
    try: ids = json.loads(b)['esearchresult']['idlist']
    except Exception: pass
    S['pmc_ids'] = ids; S['pmc_records'] = []
    if ids:
        st, b = get(EB + 'esummary.fcgi?' + urllib.parse.urlencode({'db': 'pmc', 'id': ','.join(ids), 'retmode': 'json'}), f'{cd}_pmc_esummary.json')
        try:
            res = json.loads(b)['result']
            for i in res.get('uids', []):
                x = res[i]; aid = {a.get('idtype'): a.get('value') for a in x.get('articleids', [])}
                S['pmc_records'].append({'pmcid': 'PMC' + i, 'doi': (aid.get('doi') or '').lower(), 'pmid': aid.get('pmid'), 'title': x.get('title'), 'pubdate': x.get('pubdate'), 'epubdate': x.get('epubdate')})
        except Exception as e: S['pmc_parse_error'] = repr(e)
    pmc_prot = [r for r in S['pmc_records'] if r['doi'] == doi]
    S['pmc_protocol'] = pmc_prot
    pmcids = sorted({r['pmcid'] for r in pmc_prot} | ({S['pubmed_protocol']['pmc']} if S['pubmed_protocol'] and S['pubmed_protocol'].get('pmc') else set()))
    S['protocol_identified'] = bool(S['pubmed_protocol'] or pmc_prot); S['fulltext'] = []
    for pmc in pmcids:
        st, b = get(f'https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/', f'{cd}_{pmc}_pmc.html'); S['fulltext'].append({'pmcid': pmc, 'source': 'pmc_html', 'status': st, 'bytes': len(b), 'file': f'{cd}_{pmc}_pmc.html'})
        st, b = get(EU + f'{pmc}/fullTextXML', f'{cd}_{pmc}_europepmc_fullTextXML.xml'); S['fulltext'].append({'pmcid': pmc, 'source': 'europepmc_fullTextXML', 'status': st, 'bytes': len(b), 'file': f'{cd}_{pmc}_europepmc_fullTextXML.xml'})
    out['reviews'][cd] = S
    print(cd, 'protocol', S['protocol_identified'], 'PMC records', [(r['pmcid'], r['doi'][-9:], r['pubdate']) for r in S['pmc_records']], 'fulltext', [(f['source'], f['status'], f['bytes']) for f in S['fulltext']], flush=True)
out['finished'] = now()
json.dump(out, open(os.path.join(HERE, 'stageA2_pmc.json'), 'w'), indent=1)
