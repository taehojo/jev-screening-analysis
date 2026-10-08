#!/usr/bin/env python
# npj AN-0001-13 step 1: download the pre-specified model (ncbi/MedCPT-Cross-Encoder) into this folder and record its revision.
# No substitute model is chosen if this fails (plan of 2026-09-28 19:48:38 EDT).
import os, json, subprocess, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ['HF_HOME'] = os.path.join(HERE, 'hf_home')
from huggingface_hub import snapshot_download, HfApi
import huggingface_hub
def now(): return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S %Z'], capture_output=True, text=True).stdout.strip()
REPO = 'ncbi/MedCPT-Cross-Encoder'
out = {'repo': REPO, 'started': now(), 'huggingface_hub': huggingface_hub.__version__}
info = HfApi().model_info(REPO)
out['revision_sha'] = info.sha; out['last_modified'] = str(getattr(info, 'last_modified', None)); out['files_in_repo'] = [s.rfilename for s in info.siblings]
path = snapshot_download(REPO, revision=info.sha, cache_dir=os.path.join(HERE, 'hf_home', 'hub'))
out['local_path'] = path
out['files'] = {}
for f in sorted(os.listdir(path)):
    p = os.path.join(path, f)
    if os.path.isfile(p):
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest(); out['files'][f] = {'bytes': os.path.getsize(p), 'sha256': h}
out['finished'] = now()
json.dump(out, open(os.path.join(HERE, 'model_download.json'), 'w'), indent=1)
if os.path.exists(os.path.join(path, 'README.md')):
    open(os.path.join(HERE, 'model_card_README.md'), 'w').write(open(os.path.join(path, 'README.md')).read())
print(json.dumps(out, indent=1))
