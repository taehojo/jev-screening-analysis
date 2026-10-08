# AN-0001-19: search date limits recorded in the 31 CLEF 2019 Task 2 test topic files (public data, read only); writes clef2019_topic_date_limits.json here.
import glob, re, os, json
rows = []
for f in sorted(glob.glob('/N/project/AiLab/jev/clef/tar/2019-TAR/Task2/Testing/*/topics/*')):
    t = open(f, errors='replace').read()
    cd = re.search(r'Topic:\s*(CD\d+)', t).group(1); typ = f.split('/')[-3]
    lim = [l.strip() for l in re.findall(r'limit[^\n]*', t, re.I)]
    dates = re.findall(r'(?:ED|DT|EM|DP|PD|yr)\s*=\s*"?(\d{8}|\d{4})\s*-\s*"?(\d{8}|\d{4})', t, re.I)
    rows.append({'topic': cd, 'type': typ, 'limit_lines': lim, 'date_ranges': dates})
ends = sorted((b, r['topic']) for r in rows for a, b in r['date_ranges'] if len(b) == 8)
summary = {'n_topics': len(rows), 'n_topics_with_8digit_end_date': len(ends), 'earliest_end_date': ends[0], 'latest_end_date': ends[-1], 'topics_with_year_only_limit': [(r['topic'], r['date_ranges']) for r in rows if r['date_ranges'] and len(r['date_ranges'][0][1]) == 4], 'topics_ending_20170630': [t for d, t in ends if d == '20170630'], 'source': 'clef/tar/2019-TAR/Task2/Testing/*/topics/* (Ovid MEDLINE entry-date limits in each Boolean query)'}
json.dump({'summary': summary, 'topics': rows}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clef2019_topic_date_limits.json'), 'w'), indent=1)
print(json.dumps(summary, indent=1))
