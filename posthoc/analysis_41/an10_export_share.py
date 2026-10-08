# AN-0001-10 (npj round 1): per-review share of the SYNERGY+ release records and included studies retained in the
# exported subset, and whether threshold outcomes and the Jev AUC vary with it. Post hoc; no model call.
# The release is read as files (labels.csv per review), not through the synergy-dataset package.
# The counting block is copied from review_pipeline/final_candidate_NOT_ACCEPTED/audit/part_A/work/count_synergy_raw.py
# (Lancet final audit); per-review output and the correlation analysis are added. Label column: label_included.
import csv, os, json, datetime
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
S = '/N/u/tjo/Quartz/.synergy_dataset_source/synergy-dataset-plus/'
SYN = '/N/project/AiLab/jev/synergy/'
TAU = 0.07
# ---- copied counting block (unchanged logic) ----
rs = {r['review']: r for r in csv.DictReader(open(SYN + 'review_summary.csv'))}
tot_n = tot_i = 0; used_n = used_i = 0; rows = []
for k in sorted(os.listdir(S)):
    f = S + k + '/labels.csv'
    if not os.path.exists(f): continue
    R = list(csv.DictReader(open(f)))
    key = [c for c in R[0].keys() if 'label' in c.lower() and 'included' in c.lower()]
    lab = key[0] if key else list(R[0].keys())[-1]
    n = len(R); i = sum(1 for r in R if str(r.get(lab)).strip() in ('1', '1.0', 'True'))
    tot_n += n; tot_i += i
    if k in rs: used_n += n; used_i += i; rows.append((k, n, i, int(rs[k]['n']), int(rs[k]['incl'])))
    else: rows.append((k, n, i, None, None))
chk = {'release_reviews': len(rows), 'release_records': tot_n, 'release_incl': tot_i, 'used114_release_records': used_n, 'used114_release_incl': used_i,
       'used114_local_records': sum(int(r['n']) for r in rs.values()), 'used114_local_incl': sum(int(r['incl']) for r in rs.values()), 'label_col': lab,
       'heldout_release_records': sum(r[1] for r in rows if r[0] in rs and rs[r[0]]['split'] == 'test'), 'heldout_release_incl': sum(r[2] for r in rows if r[0] in rs and rs[r[0]]['split'] == 'test')}
stored = {'used114_release_records': 478478, 'used114_release_incl': 10679, 'heldout_release_records': 115054, 'heldout_release_incl': 1493}
chk['reproduces_stored_totals'] = all(chk[k] == v for k, v in stored.items())
if not chk['reproduces_stored_totals']: print('WARNING: stored totals not reproduced', {k: (chk[k], v) for k, v in stored.items()})
# ---- per review ----
REL = {r[0]: (r[1], r[2]) for r in rows}
def per_review(recf, jevf):
    R = json.load(open(recf)); J = {d['id']: d['p'] for d in json.load(open(jevf)) if d.get('ok') and d.get('p') is not None}
    by = {}
    for r in R: by.setdefault(r['review'], []).append(r)
    out = []
    for k, v in by.items():
        if not all(r['id'] in J for r in v): continue
        y = np.array([int(r['label']) for r in v]); p = np.array([float(J[r['id']]) for r in v]); s = p >= TAU
        rn, ri = REL[k]
        out.append({'review': k, 'split': rs[k]['split'], 'export_records': len(y), 'export_included': int(y.sum()), 'release_records': rn, 'release_included': ri,
                    'share_records': len(y) / rn, 'share_included': int(y.sum()) / ri if ri else None,
                    'recall_at_tau': float(y[s].sum() / y.sum()), 'work_at_tau': float(s.mean()), 'jev_auc': float(roc_auc_score(y, p))})
    return pd.DataFrame(out).sort_values('review')
H = per_review(SYN + 'test_records.json', SYN + 'jev_test.json'); D = per_review(SYN + 'dev2000_records.json', SYN + 'jev_dev2000.json')
H.to_csv('per_review_heldout.csv', index=False); D.to_csv('per_review_development73.csv', index=False)
def describe(T):
    return {'n_reviews': len(T), 'share_records_median': float(T.share_records.median()), 'share_records_range': [float(T.share_records.min()), float(T.share_records.max())],
            'share_records_iqr': [float(T.share_records.quantile(.25)), float(T.share_records.quantile(.75))],
            'share_included_median': float(T.share_included.median()), 'share_included_range': [float(T.share_included.min()), float(T.share_included.max())],
            'share_included_iqr': [float(T.share_included.quantile(.25)), float(T.share_included.quantile(.75))],
            'totals': {'export_records': int(T.export_records.sum()), 'release_records': int(T.release_records.sum()), 'export_included': int(T.export_included.sum()), 'release_included': int(T.release_included.sum())},
            'reviews_with_recall_below_1': int((T.recall_at_tau < 1).sum()), 'reviews_with_recall_below_0.95': int((T.recall_at_tau < 0.95).sum())}
def corr(T):
    res = {}
    for sh in ('share_records', 'share_included'):
        for oc in ('recall_at_tau', 'work_at_tau', 'jev_auc'):
            r = spearmanr(T[sh], T[oc]); res[f'{sh}~{oc}'] = {'rho': float(r.correlation) if np.isfinite(r.correlation) else None, 'p': float(r.pvalue) if np.isfinite(r.pvalue) else None, 'n': len(T)}
    return res
out = {'analysis_id': 'AN-0001-10', 'release_path': S, 'reproduction_check': chk, 'heldout': {'describe': describe(H), 'spearman': corr(H)}, 'development73': {'describe': describe(D), 'spearman': corr(D)},
       'note': 'Spearman correlation, scipy.stats.spearmanr default (P from the t distribution). Recall at tau is 1 in most held-out reviews, so its variation is limited.',
       'finished': datetime.datetime.now().astimezone().isoformat(timespec='seconds')}
json.dump(out, open('results.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
