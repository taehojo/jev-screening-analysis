#!/usr/bin/env python3
"""AN-0002-01 (manuscript analysis 47, post hoc): when were the three Jev scoring modes of the 2002-record
held-out subset scored? Reads only file times (statx via GNU stat), the first and last lines of the scoring
logs (progress lines: counts and cost, no score or label), the chain script and run_screen.mjs (source text),
the key names of the output files (not their values), and the session record of the original project
(entries dated 24 and 25 September 2026 only; entries of the revision pipeline, from 26 September on, are
not read). Writes results.json. No network access, no score or label read."""
import json, pathlib, re, subprocess

SYN = pathlib.Path('/N/project/AiLab/jev/synergy')
D = pathlib.Path('/N/project/AiLab/jev/review_pipeline_npj/rounds/round_0002/revision/analysis/AN-0002-01')
SESSION = pathlib.Path('/N/u/tjo/Quartz/.claude/projects/-N-project-AiLab-jev/313fa455-92e3-4d0a-b637-fc65a2872058.jsonl')

def st(name):
    out = subprocess.run(['stat', '-c', '%w|%y|%s', str(SYN / name)], capture_output=True, text=True, check=True).stdout.strip()
    b, m, s = out.split('|')
    return {'file': f'synergy/{name}', 'birth': b[:19] + ' EDT', 'modified': m[:19] + ' EDT', 'bytes': int(s)}

def log_ends(name):
    L = (SYN / name).read_text().splitlines()
    return {'first_line': L[0], 'last_line': L[-1], 'n_lines': len(L),
            'completion_lines': [x for x in L if x.startswith('완료')], 'header_lines': [x for x in L if ' batch=' in x]}

files = {k: st(k) for k in ['jev_testcmp_single.json', 'run_p2_single.log', 'jev_testcmp_b10.json', 'run_p2_b10.log',
                            'run_p2.done', 'jev_test.json', 'run_jev_test.log', 'run_jev_q1.log', 'chain_jev.log',
                            'test_cmp_ids.json']}
logs = {k: log_ends(k) for k in ['run_p2_single.log', 'run_p2_b10.log', 'run_jev_test.log']}
chain = (SYN / 'chain_jev.sh').read_text().splitlines()
chain_test_lines = [x for x in chain if 'jev_test.json' in x]
src = (SYN / 'run_screen.mjs').read_text().splitlines()
src_save = [f'{i + 1}: {x.strip()}' for i, x in enumerate(src) if 'save()' in x or 'const save' in x]
src_provider = [f'{i + 1}: {x.strip()[:160]}' for i, x in enumerate(src) if 'provider' in x]
keys = {k: sorted(json.load(open(SYN / k))[0].keys()) for k in ['jev_testcmp_single.json', 'jev_testcmp_b10.json', 'jev_test.json']}

# ---- session record of the original project (24 and 25 September only)
sess = {}
with open(SESSION) as fh:
    for line in fh:
        if 'timestamp' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        ts = o.get('timestamp', '')
        if not (ts.startswith('2026-09-24') or ts.startswith('2026-09-25')):
            continue
        c = o.get('message', {}).get('content')
        if o.get('type') == 'assistant' and isinstance(c, list):
            for x in c:
                if x.get('type') == 'tool_use':
                    cmd = x.get('input', {}).get('command', '')
                    if 'jev_testcmp_single.json' in cmd and 'nohup' in cmd and 'launch_subset_runs' not in sess:
                        m = re.search(r'nohup bash -c "(.*?)" > /dev/null', cmd, re.S)
                        sess['launch_subset_runs'] = {'utc': ts, 'command': m.group(1) if m else None}
        s = json.dumps(c, ensure_ascii=False) if c is not None else ''
        if o.get('type') == 'user':
            if 'full_run_progress_check' not in sess and re.search(r'\d+/33001 ', s):
                dates = re.findall(r'[A-Z][a-z]{2} Sep 25 \d\d:\d\d:\d\d EDT 2026', s)
                prog = re.findall(r'(\d+/33001)  ', s)
                sess['full_run_progress_check'] = {'utc': ts, 'date_outputs': dates, 'progress': prog[:4]}
            if 'full_run_after_retry' not in sess and 'jev_test n 33001' in s:
                sess['full_run_after_retry'] = {'utc': ts, 'date_outputs': re.findall(r'[A-Z][a-z]{2} Sep 25 \d\d:\d\d:\d\d EDT 2026', s),
                                                'lines': re.findall(r'(jev batch=10: 33001[^"\\]*|jev_test n 33001 ok 33001)', s)}
            if 'subset_runs_completed' not in sess and '완료. 비용 $0.0845' in s and '완료. 비용 $0.0404' in s:
                sess['subset_runs_completed'] = {'utc': ts, 'lines': re.findall(r'완료\. 비용 \$0\.0(?:845|404)[^"\\]*', s)}

res = {'analysis': 'AN-0002-01', 'manuscript_analysis_number': 47, 'post_hoc': True, 'timezone': 'US Eastern (EDT, UTC-4)',
       'files': files, 'logs': logs, 'chain_jev_sh_lines_for_jev_test': chain_test_lines,
       'run_screen_save_lines': src_save, 'run_screen_provider_lines': src_provider, 'output_file_keys': keys,
       'session_record': {'file': str(SESSION), 'scope': 'entries with timestamps on 2026-09-24 and 2026-09-25 (original project work)', **sess}}

res['modes'] = {
    'single_record_requests': {
        'output': 'synergy/jev_testcmp_single.json', 'log': 'synergy/run_p2_single.log',
        'start': '2026-09-25 09:15:48 EDT',
        'start_evidence': 'launch command in the session record at 13:15:48 UTC (09:15:48 EDT); birth time of run_p2_single.log 09:15:48',
        'end': '2026-09-25 10:28:38 EDT',
        'end_evidence': 'last write of jev_testcmp_single.json and run_p2_single.log 10:28:38 (log ends with the completion line); the next command of the same sequential chain created run_p2_b10.log at 10:28:38',
        'requests': logs['run_p2_single.log']['header_lines']},
    'ten_record_batches_formed_within_subset': {
        'output': 'synergy/jev_testcmp_b10.json', 'log': 'synergy/run_p2_b10.log',
        'start': '2026-09-25 10:28:38 EDT',
        'start_evidence': 'birth time of run_p2_b10.log 10:28:38; the command followed the single-record command in the same bash -c chain (session record)',
        'end': '2026-09-25 10:39:29 EDT',
        'end_evidence': 'last write of jev_testcmp_b10.json and run_p2_b10.log 10:39:29 (log ends with the completion line); run_p2.done written 10:39:29 by the chain; completion lines seen in the session record at 14:40:57 UTC',
        'requests': logs['run_p2_b10.log']['header_lines']},
    'ten_record_batches_of_full_heldout_run': {
        'output': 'synergy/jev_test.json', 'log': 'synergy/run_jev_test.log',
        'start': '2026-09-25 00:48:48 EDT',
        'start_evidence': 'birth time of run_jev_test.log 00:48:48 (created by the redirection of the command in chain_jev.sh); session record at 00:49:50 EDT shows 200 of 33,001 records done; same start as SI Note 2 of v0001',
        'end': 'at or before 2026-09-25 03:29:32 EDT (exact end of the scoring invocation not recorded)',
        'end_evidence': 'chain_jev.sh ran a second, retry invocation into the same log and file; the log shows that it found 33,001 of 33,001 records done and sent 0 requests, and run_screen.mjs writes the output file at the end of every invocation, so the last write of jev_test.json and run_jev_test.log (03:29:32) is the end of the retry; the next chain command created run_jev_q1.log at 03:29:32; session record at 03:29:40 EDT confirms 33,001 ok',
        'when_the_2002_subset_records_were_scored_within_the_run': 'not recorded (no per-record time in the output or the log)',
        'requests': logs['run_jev_test.log']['header_lines']}}
res['same_day_batched_modes'] = True
res['gap_between_full_run_end_and_subset_batches_start'] = 'about 7 hours (end at or before 03:29:32, start 10:28:38 on 25 September 2026)'
res['provider_recorded'] = False
res['provider_evidence'] = 'run_screen.mjs reads provider_metadata only to add the gateway cost (line with marketCost); the stored records have only the keys listed in output_file_keys, none of which names the provider'
json.dump(res, open(D / 'results.json', 'w'), indent=1, ensure_ascii=False)
for k, v in res['modes'].items():
    print(k, '|', v['start'], '->', v['end'])
print('session', json.dumps(sess, ensure_ascii=False)[:1500])
print('keys', keys)
