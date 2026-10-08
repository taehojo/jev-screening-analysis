#!/usr/bin/env python
# npj AN-0001-15 stage A decision: per review, whether a protocol version was identified and whether its retrieved full text
# contains the sections "Objectives" and "Criteria for considering studies for this review" (headings of the saved PMC page).
import os, re, json, html, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(HERE, 'sources')
def now(): return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S %Z'], capture_output=True, text=True).stdout.strip()
A = json.load(open(os.path.join(HERE, 'stageA_retrieval.json'))); A2 = json.load(open(os.path.join(HERE, 'stageA2_pmc.json')))
rows = {}
for cd, R in A['reviews'].items():
    S = A2['reviews'][cd]; ft = [f for f in S['fulltext'] if f['source'] == 'pmc_html' and f['status'] == 200]
    sec = {'objectives': False, 'criteria': False, 'page_is_recaptcha_challenge': None}
    for f in ft:
        s = open(os.path.join(SRC, f['file']), encoding='utf-8', errors='ignore').read()
        heads = [re.sub(r'<[^>]+>', '', h).strip().lower() for h in re.findall(r'<h[234][^>]*>(.*?)</h[234]>', s, flags=re.S)]
        sec = {'objectives': 'objectives' in heads, 'criteria': 'criteria for considering studies for this review' in heads, 'page_is_recaptcha_challenge': 'RecaptchaChallengePage' in s}
    rows[cd] = {'n_records': R['n_records'], 'cdsr_versions_in_pubmed': [(v['doi'], v['year']) for v in R['pubmed_versions']],
                'unsuffixed_doi_in_pubmed_is_review': [v['year'] for v in R['pubmed_versions'] if v['doi'] == R['protocol_doi'].lower() and not re.search(r'this is (a|the) protocol', v['abstract_start'] or '', re.I)],
                'protocol_identified': S['protocol_identified'], 'protocol_pmcid': [f['pmcid'] for f in S['fulltext']][:1],
                'fulltext_pmc_html_status': [f['status'] for f in S['fulltext'] if f['source'] == 'pmc_html'],
                'fulltext_europepmc_xml_status': [f['status'] for f in S['fulltext'] if f['source'] == 'europepmc_fullTextXML'],
                'sections_found': sec, 'usable': bool(S['protocol_identified'] and sec['objectives'] and sec['criteria']),
                'cochrane_library_status': R['cochrane_library_status'], 'wiley_status': R['wiley_status']}
n_id = sum(r['protocol_identified'] for r in rows.values()); n_use = sum(r['usable'] for r in rows.values())
out = {'written': now(), 'n_reviews': len(rows), 'protocols_identified': n_id, 'protocols_usable': n_use, 'usable_reviews': [k for k, r in rows.items() if r['usable']],
       'identified_not_usable': [k for k, r in rows.items() if r['protocol_identified'] and not r['usable']], 'threshold': 8,
       'stage_B_run': n_use >= 8, 'reviews': rows}
json.dump(out, open(os.path.join(HERE, 'stageA_decision.json'), 'w'), indent=1)
print({k: v for k, v in out.items() if k != 'reviews'})
