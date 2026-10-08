#!/usr/bin/env python
# npj AN-0001-13 step 2 (post hoc): zero-shot scores of the MedCPT Cross-Encoder for held-out SYNERGY+ (23) and CLEF 2019 (31) records.
# Query = the criteria text used by the bge-base baseline (synergy/criteria.json, clef/clef_criteria.json; as in synergy/zs_baselines.py);
# document = f"{title}. {abstract}" (as in synergy/zs_baselines.py). Pairs truncated to 512 tokens (tokenizer truncation=True, max_length=512).
# Scores are the logits of the model. Writes scores/<set>_<review>.json in this folder only (resumable). CPU only; no labels are used here.
import os, sys, json, time, subprocess, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ['HF_HOME'] = os.path.join(HERE, 'hf_home'); os.environ['HF_HUB_OFFLINE'] = '1'
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
ROOT = '/N/project/AiLab/jev'
def now(): return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S %Z'], capture_output=True, text=True).stdout.strip()
ap = argparse.ArgumentParser(); ap.add_argument('--threads', type=int, default=16); ap.add_argument('--batch', type=int, default=32); a = ap.parse_args()
torch.set_num_threads(a.threads)
D = json.load(open(os.path.join(HERE, 'model_download.json'))); path = D['local_path']
tok = AutoTokenizer.from_pretrained(path); model = AutoModelForSequenceClassification.from_pretrained(path).eval()
print('model', D['repo'], D['revision_sha'], 'torch', torch.__version__, 'threads', torch.get_num_threads(), now(), flush=True)
SETS = [('test', ROOT + '/synergy/test_records.json', ROOT + '/synergy/criteria.json'), ('clef', ROOT + '/clef/clef_records.json', ROOT + '/clef/clef_criteria.json')]
os.makedirs(os.path.join(HERE, 'scores'), exist_ok=True)
tot = 0; t00 = time.time()
for name, rp, cp in SETS:
    R = json.load(open(rp)); C = json.load(open(cp)); by = {}
    for r in R: by.setdefault(r['review'], []).append(r)
    for k in sorted(by):
        if k not in C: print('no criteria', k, flush=True); continue
        outp = os.path.join(HERE, 'scores', f'{name}_{k}.json')
        if os.path.exists(outp): continue
        recs = by[k]; q = C[k]; docs = [f"{r['title']}. {r['abstract']}" for r in recs]
        t0 = time.time(); s = []; ntrunc = 0
        with torch.no_grad():
            for i in range(0, len(docs), a.batch):
                pairs = [[q, d] for d in docs[i:i + a.batch]]
                full = tok(pairs, truncation=False, padding=False)['input_ids']; ntrunc += sum(len(x) > 512 for x in full)
                enc = tok(pairs, truncation=True, padding=True, return_tensors='pt', max_length=512)
                s += [float(x) for x in model(**enc).logits.squeeze(dim=1)]
        json.dump({'ids': [r['id'] for r in recs], 'medcpt_ce': s, 'n_pairs_truncated': ntrunc, 'query_tokens': len(tok(q)['input_ids']), 'finished': now()}, open(outp, 'w'))
        tot += len(docs); print(f'{name} {k} n={len(docs)} truncated={ntrunc} {time.time()-t0:.0f}s total={tot} {time.time()-t00:.0f}s', flush=True)
print('done', now(), flush=True)
