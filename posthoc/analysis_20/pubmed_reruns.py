# AN-0001-20: re-run of the Research in context PubMed searches (esearch) with and without the start-date limit, and comparison of the subset with manuscript/ric_subset32.json.
# Run on 2026-09-26 (see api_responses/run_timestamp.txt); responses saved in api_responses/. Uses curl because Python urllib fails TLS verification on this server.
import json, subprocess, urllib.parse, time, os
H = os.path.dirname(os.path.abspath(__file__)); A = os.path.join(H, 'api_responses'); os.makedirs(A, exist_ok=True)
base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
core = '("large language model*"[tiab] OR "LLM"[tiab] OR "LLMs"[tiab] OR "GPT-4*"[tiab] OR "ChatGPT"[tiab] OR "generative artificial intelligence"[tiab]) AND ("systematic review*"[tiab] OR "evidence synthesis"[tiab] OR "meta-analys*"[tiab]) AND ("screening"[tiab] OR "study selection"[tiab] OR "citation screening"[tiab] OR "abstract screening"[tiab])'
sub = ' AND ("active learning"[tiab] OR "prioriti*"[tiab] OR "stopping"[tiab])'
Q = {'q1_original_2022-11-30_to_2026-09-25': core + ' AND ("2022/11/30"[dp] : "2026/09/25"[dp])', 'q2_no_date_limit': core, 'q3_before_2022-11-30': core + ' AND ("1900/01/01"[dp] : "2022/11/29"[dp])',
     'q4_subset_original_dates': core + sub + ' AND ("2022/11/30"[dp] : "2026/09/25"[dp])', 'q5_subset_no_date_limit': core + sub, 'q6_subset_before_2022-11-30': core + sub + ' AND ("1900/01/01"[dp] : "2022/11/29"[dp])', 'q7_original_2022-11-30_to_2026-09-26': core + ' AND ("2022/11/30"[dp] : "2026/09/26"[dp])'}
out = {}
for k, q in Q.items():
    url = base + '?' + urllib.parse.urlencode({'db': 'pubmed', 'term': q, 'retmode': 'json', 'retmax': 500})
    r = subprocess.run(['curl', '-sS', '-m', '60', '-A', 'ldh-reviser/1.0 (mailto:[email withheld])', url], capture_output=True, text=True)
    open(f'{A}/pubmed_{k}.json', 'w').write(r.stdout)
    d = json.loads(r.stdout)['esearchresult']; out[k] = {'count': int(d['count']), 'n_ids_returned': len(d['idlist']), 'query': q}
    print(k, 'count', d['count']); time.sleep(0.5)
d = json.load(open(f'{A}/pubmed_q4_subset_original_dates.json'))['esearchresult']['idlist']
sub32 = json.load(open('/N/project/AiLab/jev/manuscript/ric_subset32.json')); pm32 = {x['pmid'] for x in sub32}
out['subset_check'] = {'ric_subset32_entries': len(sub32), 'q4_ids': len(d), 'in_q4_not_in_file': sorted(set(d) - pm32), 'in_file_not_in_q4': sorted(pm32 - set(d))}
json.dump(out, open(f'{A}/pubmed_counts_summary.json', 'w'), indent=1); print(out['subset_check'])
