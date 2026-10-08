# AN-0001-12 (npj round 1): what a single review team can expect. Random-effects meta-analysis of the logit of the
# per-review proportion read, with a 95% prediction interval for a new review; for the threshold (tau=0.07) and for the
# combined workflow (Jev ranking plus the statistical criterion of Callaghan and Mueller-Hansen), in held-out and CLEF
# reviews. Empirical percentiles are given alongside; recall at tau is described empirically only. Post hoc; no model call.
# Method: within-review variance of logit(p) = 1 / (N p (1 - p)) with p clipped to [0.5/N, 1 - 0.5/N];
# DerSimonian-Laird between-review variance; 95% CI of the pooled logit with z; 95% prediction interval
# mu +- t_{k-2, 0.975} sqrt(tau2 + SE(mu)^2) (Higgins, Thompson and Spiegelhalter 2009), back-transformed.
import json, datetime
import numpy as np, pandas as pd
from scipy.stats import t as tdist, norm
AN = '/N/project/AiLab/jev/review_pipeline/rounds/round_0001/revision/analysis'
P4 = {'heldout': pd.read_csv(AN + '/AN-0001-04/per_review_heldout.csv'), 'clef': pd.read_csv(AN + '/AN-0001-04/per_review_clef.csv')}
P8 = pd.read_csv(AN + '/AN-0001-08/per_review_stopping.csv')
def expit(x): return 1 / (1 + np.exp(-x))
def re_meta(p, N):
    p = np.asarray(p, float); N = np.asarray(N, float); pc = np.clip(p, 0.5 / N, 1 - 0.5 / N)
    y = np.log(pc / (1 - pc)); v = 1 / (N * pc * (1 - pc)); w = 1 / v; k = len(y)
    mu_fe = np.sum(w * y) / np.sum(w); Q = np.sum(w * (y - mu_fe) ** 2); C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / C); ws = 1 / (v + tau2); mu = np.sum(ws * y) / np.sum(ws); se = np.sqrt(1 / np.sum(ws))
    z = norm.ppf(0.975); tq = tdist.ppf(0.975, k - 2); half = tq * np.sqrt(tau2 + se ** 2)
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return {'k': k, 'pooled': float(expit(mu)), 'ci95': [float(expit(mu - z * se)), float(expit(mu + z * se))], 'pi95': [float(expit(mu - half)), float(expit(mu + half))],
            'tau2_logit': float(tau2), 'Q': float(Q), 'I2': float(I2), 't_quantile_df_k_minus_2': float(tq), 'n_clipped': int(np.sum((p < 0.5 / N) | (p > 1 - 0.5 / N))),
            'empirical_p5_p95': [float(np.percentile(p, 5)), float(np.percentile(p, 95))], 'empirical_min_max': [float(p.min()), float(p.max())], 'mean': float(p.mean()), 'median': float(np.median(p))}
out = {'analysis_id': 'AN-0001-12', 'inputs': {'threshold': 'Lancet AN-0001-04 per_review_{heldout,clef}.csv thr_work, thr_rec', 'combined_workflow': 'Lancet AN-0001-08 per_review_stopping.csv jev_stat_work, jev_stat_rec (mean over ten tie-break seeds)'},
       'percentiles_note': 'numpy.percentile with linear interpolation'}
for coll in ('heldout', 'clef'):
    T4 = P4[coll][P4[coll].thr_work.notna()].set_index('review'); T8 = P8[P8.collection == coll].set_index('review')
    assert set(T4.index) == set(T8.index), coll
    revs = sorted(T8.index); N = T8.loc[revs, 'N'].values
    assert np.all(T4.loc[revs, 'N'].values == N)
    thr = T4.loc[revs, 'thr_work'].values; comb = T8.loc[revs, 'jev_stat_work'].values
    rec_thr = T4.loc[revs, 'thr_rec'].values; rec_comb = T8.loc[revs, 'jev_stat_rec'].values
    out[coll] = {'threshold_proportion_read': re_meta(thr, N), 'combined_workflow_proportion_read': re_meta(comb, N),
                 'threshold_recall_empirical': {'mean': float(rec_thr.mean()), 'p5': float(np.percentile(rec_thr, 5)), 'median': float(np.median(rec_thr)), 'min': float(rec_thr.min()), 'n_equal_1': int(np.sum(rec_thr == 1)), 'n': len(rec_thr)},
                 'combined_recall_empirical': {'mean': float(rec_comb.mean()), 'p5': float(np.percentile(rec_comb, 5)), 'median': float(np.median(rec_comb)), 'min': float(rec_comb.min()), 'n_equal_1': int(np.sum(rec_comb == 1)), 'n': len(rec_comb)},
                 'difference_combined_minus_threshold_empirical_p5_p95': [float(np.percentile(comb - thr, 5)), float(np.percentile(comb - thr, 95))]}
out['finished'] = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
json.dump(out, open('results.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
