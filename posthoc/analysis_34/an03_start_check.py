# npj AN-0001-03 diagnostic: for each review and seed, the records labelled in the starting phase of the hybrid
# (Jev order with ties broken by np.random.RandomState(seed).rand, read until both classes are seen), as in asr_sim.py run().
import json, sys, numpy as np
recs, jev = sys.argv[1], sys.argv[2]
R = json.load(open(recs)); J = {d['id']: d['p'] for d in json.load(open(jev)) if d.get('ok') and d.get('p') is not None}
by = {}
for r in R: by.setdefault(r['review'], []).append(r)
out = {}
for k, rs in sorted(by.items()):
    y = np.array([r['label'] for r in rs]); 
    if not (0 < y.sum() < len(y)) or not all(r['id'] in J for r in rs): continue
    p = np.clip(np.array([J[r['id']] for r in rs], float), 1e-4, 1 - 1e-4); jl = np.log(p / (1 - p)); N = len(y)
    starts = []
    for s in range(10):
        rng = np.random.RandomState(s); perm = np.lexsort((rng.rand(N), -jl)); y2 = y[perm]
        for i in range(N):
            if y2[:i + 1].max() == 1 and y2[:i + 1].min() == 0: break
        starts.append(tuple(sorted(int(x) for x in perm[:i + 1])))
    top = np.sort(p)[::-1]; ntop = int((p == top[0]).sum())
    out[k] = {'N': N, 'start_len_by_seed': [len(t) for t in starts], 'distinct_start_sets': len(set(starts)), 'top_p': float(top[0]), 'n_records_at_top_p': ntop,
              'labels_at_top_p': [int(v) for v in np.unique(y[p == top[0]])]}
    print(k, out[k], flush=True)
json.dump(out, open(sys.argv[3], 'w'), indent=1)
