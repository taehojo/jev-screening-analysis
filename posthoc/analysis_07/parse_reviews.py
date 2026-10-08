# AN-0001-07 step 2: parse the PMC HTML pages of the three Cochrane reviews (public, fetched 2026-09-26; see sources/fetch_log.txt)
# to map every CLEF content-level relevant PMID to the review's study identifier and reference-list section,
# and to extract the review text that mentions each missed study.
import re, html, json, os
OUT = os.path.dirname(os.path.abspath(__file__)); ROOT = '/N/project/AiLab/jev'
PMC = {'CD012233': 'PMC6513652', 'CD011977': 'PMC6494477', 'CD010558': 'PMC6494651'}
def text(t): return html.unescape(' '.join(re.sub(r'<[^>]+>', ' ', t).split()))
missed = json.load(open(f'{OUT}/missed_studies.json'))
rec = {r['id'].split('|')[1]: r for r in json.load(open(f'{ROOT}/clef/clef_records.json')) if r['review'] in PMC}
jev = {d['id']: d['p'] for d in json.load(open(f'{ROOT}/synergy/jev_clef.json')) if d.get('ok')}
out = {}
for cd, pmc in PMC.items():
    t = open(f'{OUT}/sources/{pmc}.html', encoding='utf-8', errors='replace').read()
    # reference-list sections: heading text followed by <section class="ref-list"> blocks with <li id=...> entries
    secs = {}
    heads = [(m.start(), text(m.group(1))) for m in re.finditer(r'<h3[^>]*>(.*?)</h3>', t, flags=re.S)]
    ref_heads = [(pos, h) for pos, h in heads if h.startswith('References to') or h == 'Additional references' or h.startswith('References to other')]
    for i, (pos, h) in enumerate(ref_heads):
        end = ref_heads[i + 1][0] if i + 1 < len(ref_heads) else t.find('<h2', pos + 10)
        block = t[pos:end]
        # study entries: heading like "Chen 1997 {published data only}" then citations; PMIDs from pubmed links
        entries = re.split(r'(?=<section id="[^"]*-bbs2-\d+" class="ref-list">)', block)
        studies = []
        for e in entries:
            m = re.search(r'class="ref-list">\s*(?:<[^>]+>\s*)*([^<]{2,80}?)\s*(?:\{|<)', e)
            if not m: continue
            sid = m.group(1).strip().rstrip('.').strip(); pmids = sorted(set(re.findall(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', e)))
            cites = [text(c) for c in re.findall(r'<li[^>]*>(.*?)</li>', e, flags=re.S)]
            studies.append({'study_id': sid, 'pmids': pmids, 'n_citations': len(cites), 'citations': [c[:300] for c in cites]})
        secs[h] = studies
    out[cd] = {'pmc': pmc, 'sections': {h: len(v) for h, v in secs.items()}, 'studies': secs}
    # map CLEF relevant PMIDs of this review to study ids
    rel = [p for p, r in rec.items() if r['review'] == cd and int(r['label'] or 0) == 1]
    pm2 = {}
    for h, studies in secs.items():
        for s in studies:
            for p in s['pmids']: pm2.setdefault(p, []).append((h, s['study_id']))
    out[cd]['clef_relevant_pmids'] = {p: {'p': jev[f'{cd}|{p}'], 'found_in': pm2.get(p, []), 'title': rec[p]['title'][:120]} for p in rel}
    out[cd]['n_relevant_not_found_in_reference_list'] = sum(1 for p in rel if p not in pm2)
    out[cd]['included_study_ids'] = [s['study_id'] for s in secs.get('References to studies included in this review', [])]
    # for each missed study: study id, other reports of the same study id and their Jev p, and sentences in the review text mentioning the study id
    body = text(t)
    for m in missed['reviews'][cd]['missed']:
        hits = pm2.get(m['pmid'], []); m['reference_list_sections'] = hits
        sid = hits[0][1] if hits else None; m['study_id'] = sid
        if sid:
            same = [s for h, studies in secs.items() for s in studies if s['study_id'] == sid]
            m['other_pmids_same_study'] = [{'pmid': p, 'in_clef_set': (f'{cd}|{p}' in jev), 'p': jev.get(f'{cd}|{p}'), 'clef_label': (int(rec[p]['label'] or 0) if p in rec else None)} for s in same for p in s['pmids'] if p != m['pmid']]
            m['study_citations'] = [c for s in same for c in s['citations']]
            sents = [s_ for s_ in re.split(r'(?<=[.!?])\s+(?=[A-Z(])', body) if re.search(re.escape(sid), s_)]
            m['review_text_mentions'] = [s_[:700] for s_ in sents[:40]]; m['n_review_text_mentions'] = len(sents)
            # characteristics of included studies entry
            k = body.find(f'{sid}. Methods') if body.find(f'{sid}. Methods') > 0 else body.find(f'{sid} Methods')
            if k < 0:
                mm = re.search(re.escape(sid) + r'\.?\s+Methods', body); k = mm.start() if mm else -1
            m['characteristics_entry'] = body[k:k + 2500] if k >= 0 else None
    out[cd]['missed'] = missed['reviews'][cd]['missed']
json.dump(out, open(f'{OUT}/review_parse.json', 'w'), indent=1, ensure_ascii=False)
for cd, v in out.items():
    print('=====', cd, v['pmc'], v['sections']); print(' included study ids (%d):' % len(v['included_study_ids']), v['included_study_ids'])
    print(' CLEF relevant PMIDs not found in reference list:', v['n_relevant_not_found_in_reference_list'])
    for p, d in sorted(v['clef_relevant_pmids'].items(), key=lambda x: x[1]['p']): print('   ', p, d['p'], d['found_in'], '|', d['title'][:80])
    for m in v['missed']:
        print(' MISSED', m['pmid'], m['p'], m.get('study_id'), m['reference_list_sections']); print('   other reports:', m.get('other_pmids_same_study')); print('   mentions:', m.get('n_review_text_mentions'))
        for s_ in (m.get('review_text_mentions') or [])[:40]: print('     -', s_[:400])
        print('   characteristics:', (m.get('characteristics_entry') or '')[:1500])
