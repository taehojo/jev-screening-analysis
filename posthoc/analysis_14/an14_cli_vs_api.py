# AN-0001-14: compare the ASReview 3.0.8 command-line simulation with the programming-interface simulation used in asr_sim.py (asr_prior method).
# The API path below copies the asr_prior branch of synergy/asr_sim.py run() line by line (perm = arange, Tfidf 1-2gram, SVM C=0.11, Balanced 9.8, Max, LastRelevant,
# one included and one excluded prior chosen with np.random.RandomState(seed)) and returns the labelling order instead of metrics. The original file is not modified.
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[_v] = '1'
import json, sys, subprocess, sqlite3, zipfile, tempfile, time, warnings, numpy as np, pandas as pd, scipy.sparse as sp
warnings.filterwarnings('ignore')
from asreview import Simulate, ActiveLearningCycle
from asreview.models.queriers import Max
from asreview.models.stoppers import LastRelevant
from asreview.models.balancers import Balanced
from asreview.models.classifiers import SVM
from asreview.models.feature_extractors import Tfidf
import asreview
OUT = os.path.dirname(os.path.abspath(__file__)); S = '/N/project/AiLab/jev/synergy/'
REVIEWS = ['Theobald_2021', 'Abgaz_2023']; SEEDS = [0, 1, 2]
R = json.load(open(S + 'test_records.json'))
res = {'asreview_version': asreview.__version__, 'reviews': REVIEWS, 'seeds': SEEDS, 'cases': []}
def api_order(df, y, seed, global_seed=None):
    if global_seed is not None: np.random.seed(global_seed)
    N = len(y)
    X = sp.csr_matrix(Tfidf(ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_df=0.95).fit_transform(df))
    clf = SVM(C=0.11, loss='squared_hinge')
    cycles = [ActiveLearningCycle(querier=Max(), classifier=clf, balancer=Balanced(ratio=9.8))]
    sim = Simulate(X, y, cycles, stopper=LastRelevant(), skip_transform=True, print_progress=False)
    sim.label([int(np.random.RandomState(seed).choice(np.where(y == 1)[0], 1, replace=False)[0])])
    sim.label([int(np.random.RandomState(seed).choice(np.where(y == 0)[0], 1, replace=False)[0])])
    sim.review()
    return [int(i) for i in sim._results['record_id'].values]
def cli_order(csv_path, seed, out_path):
    if os.path.exists(out_path): os.remove(out_path)
    cmd = [S + '.venv/bin/asreview', 'simulate', csv_path, '-o', out_path, '--n-prior-included', '1', '--n-prior-excluded', '1', '--prior-seed', str(seed), '--seed', str(seed)]
    t0 = time.time(); p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, 'PYTHONNOUSERSITE': '1'}); dt = time.time() - t0
    open(out_path + '.stdout.txt', 'w').write(p.stdout + '\n--- stderr ---\n' + p.stderr)
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(out_path) as z: z.extractall(td)
        dbs = [os.path.join(dp, f) for dp, dn, fn in os.walk(td) for f in fn if f.endswith('.sqlite') or f.endswith('.db')]
        con = sqlite3.connect(dbs[0]); rows = con.execute('SELECT rowid, record_id, label FROM results ORDER BY rowid').fetchall(); con.close()
    return [int(r[1]) for r in rows], [int(r[2]) for r in rows], ' '.join(cmd), dt, p.returncode
for k in REVIEWS:
    rs = [r for r in R if r['review'] == k]; y = np.array([r['label'] for r in rs]); df = pd.DataFrame({'title': [r['title'] for r in rs], 'abstract': [r['abstract'] for r in rs]})
    csv_path = f'{OUT}/{k}.csv'; pd.DataFrame({'title': df.title, 'abstract': df.abstract, 'included': y}).to_csv(csv_path, index=False)
    for s in SEEDS:
        a = api_order(df, y, s)
        c, clab, cmd, dt, rc = cli_order(csv_path, s, f'{OUT}/{k}_seed{s}.asreview')
        n = min(len(a), len(c)); ident = int(sum(1 for i in range(n) if a[i] == c[i])); first_diff = next((i for i in range(n) if a[i] != c[i]), None)
        case = {'review': k, 'N': len(y), 'n_included': int(y.sum()), 'seed': s, 'cli_command': cmd, 'cli_returncode': rc, 'cli_seconds': round(dt, 1), 'api_positions': len(a), 'cli_positions': len(c), 'positions_compared': n, 'identical_positions': ident, 'first_difference_at': first_diff,
                'cli_labels_match_dataset': bool(all(clab[i] == y[c[i]] for i in range(len(c)))), 'api_last_relevant_rank': len(a), 'cli_last_relevant_rank': len(c), 'prior_records_api': a[:2], 'prior_records_cli': c[:2]}
        if ident != n or len(a) != len(c):
            a2 = api_order(df, y, s, global_seed=s); n2 = min(len(a2), len(c)); case['api_with_global_seed_identical_positions'] = int(sum(1 for i in range(n2) if a2[i] == c[i])); case['api_with_global_seed_positions'] = len(a2)
        pd.DataFrame({'position': range(1, n + 1), 'api_record_id': a[:n], 'cli_record_id': c[:n], 'api_label': [int(y[i]) for i in a[:n]], 'cli_label': [int(y[i]) for i in c[:n]]}).to_csv(f'{OUT}/order_{k}_seed{s}.csv', index=False)
        res['cases'].append(case); print(json.dumps(case), flush=True)
res['summary'] = {'n_reviews': len(REVIEWS), 'n_cases': len(res['cases']), 'positions_compared_total': int(sum(c['positions_compared'] for c in res['cases'])), 'identical_total': int(sum(c['identical_positions'] for c in res['cases'])),
                  'all_identical': bool(all(c['identical_positions'] == c['positions_compared'] and c['api_positions'] == c['cli_positions'] for c in res['cases']))}
json.dump(res, open(OUT + '/results.json', 'w'), indent=1); print(json.dumps(res['summary'], indent=1))
