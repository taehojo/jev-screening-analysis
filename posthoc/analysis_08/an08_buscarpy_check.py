#!/usr/bin/env python
# AN-0001-08: cross-check of the criterion implementation in an08_stopping.py against buscarpy 0.0.2 (calculate_h0), run in the
# separate environment review_pipeline/.venv_buscarpy. Input: stat_p_curves_hybrid.json (hybrid order per review: labels in
# screening order, evaluation points and the p values computed by an08_stopping.py). For every review, buscarpy is evaluated at up
# to 60 evenly spaced evaluation points plus the first triggering point and its predecessor (buscarpy is slow on long sequences).
import os, sys, json, time, subprocess
import numpy as np, pandas as pd
from multiprocessing import Pool
import buscarpy
HERE = os.path.dirname(os.path.abspath(__file__))
def now(): return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S %Z'], capture_output=True, text=True).stdout.strip()

def one(args):
    key, N, labels, sched, p_fast = args
    labels = np.asarray(labels, np.int8); sched = np.asarray(sched); p_fast = np.asarray(p_fast, float)
    idx = np.unique(np.concatenate([np.linspace(0, len(sched) - 1, min(60, len(sched))).round().astype(int), np.where(p_fast < 0.05)[0][:1], np.maximum(np.where(p_fast < 0.05)[0][:1] - 1, 0)]))
    rows = []; t0 = time.time()
    for j in idx:
        t = int(sched[j]); pb = buscarpy.calculate_h0(labels[:t], N=N, recall_target=0.95)
        pb = float(pb) if pb is not None else float('nan')
        rows.append({'key': key, 'N': N, 't': t, 'p_fast': float(p_fast[j]), 'p_buscarpy': pb, 'nan_buscarpy': int(np.isnan(pb)), 'abs_diff': abs(float(p_fast[j]) - pb) if not np.isnan(pb) else float('nan'),
                     'same_decision': int(np.isnan(pb) or ((pb < 0.05) == (p_fast[j] < 0.05)))})
    return rows, time.time() - t0

def main():
    started = now(); print('buscarpy check started', started, 'buscarpy version', getattr(buscarpy, '__version__', 'n/a'), flush=True)
    P = json.load(open(os.path.join(HERE, 'stat_p_curves_hybrid.json')))
    tasks = [(k, v['N'], v['labels_in_order'], v['schedule'], v['p']) for k, v in P.items()]
    tasks.sort(key=lambda t: -t[1])
    with Pool(24) as pool: outs = pool.map(one, tasks, chunksize=1)
    rows = [r for o, _ in outs for r in o]; T = pd.DataFrame(rows)
    T.to_csv(os.path.join(HERE, 'buscarpy_check.csv'), index=False)
    ok = T[T.nan_buscarpy == 0]
    summ = {'started': started, 'finished': now(), 'n_points': int(len(T)), 'n_reviews': int(T.key.nunique()), 'n_nan_buscarpy': int(T.nan_buscarpy.sum()),
            'max_abs_diff_non_nan': float(ok.abs_diff.max()) if len(ok) else None, 'n_decision_mismatch': int((T.same_decision == 0).sum()),
            'seconds_buscarpy_total': float(sum(s for _, s in outs)),
            'pip_freeze': open(os.path.join(HERE, 'buscarpy_venv_freeze.txt')).read().split() if os.path.exists(os.path.join(HERE, 'buscarpy_venv_freeze.txt')) else None,
            'note': 'buscarpy.calculate_h0 returns nan when the smallest look-back window (the last screened record alone) has Ktar larger than its population (python min() over an array whose first element is nan); an08_stopping.py assigns p = 0 to such windows because H0 cannot hold there.'}
    json.dump(summ, open(os.path.join(HERE, 'buscarpy_check.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != 'pip_freeze'}, indent=1), flush=True)

if __name__ == '__main__':
    main()
