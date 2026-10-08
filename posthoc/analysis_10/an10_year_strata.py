# AN-0001-10: discrimination and threshold recall by review publication year and label publicity; same-label-level comparison for the pilot review.
import json, os, re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
S = '/N/project/AiLab/jev/synergy/'; OUT = os.path.dirname(os.path.abspath(__file__)); TAU = 0.07
res = {'tau': TAU}
def year_of(key):
    m = re.search(r'(\d{4})[a-z]?$', key); return int(m.group(1))
def per_review(records, scores, label_key='label', zs_prefix=None):
    by = {}
    for r in records: by.setdefault(r['review'], []).append(r)
    rows = []
    for k, v in by.items():
        if label_key == 'label_ta' and any(r.get('label_ta') is None for r in v): continue
        if not all(r['id'] in scores for r in v): continue
        y = np.array([r[label_key] for r in v]); p = np.array([scores[r['id']] for r in v])
        if not (0 < y.sum() < len(y)): continue
        s = p >= TAU
        row = {'review': k, 'n': len(v), 'n1': int(y.sum()), 'jev_auc': roc_auc_score(y, p), 'tau_recall': float(y[s].sum() / y.sum()), 'tau_work': float(s.mean())}
        if zs_prefix and os.path.exists(f'{S}scores_zs/{zs_prefix}_{k}.json'):
            Z = json.load(open(f'{S}scores_zs/{zs_prefix}_{k}.json'))
            for m in ('bge', 'minilm'):
                if m in Z: row[f'{m}_auc'] = roc_auc_score(y, Z[m])
        rows.append(row)
    return pd.DataFrame(rows)
R_test = json.load(open(S + 'test_records.json')); J_test = {d['id']: d['p'] for d in json.load(open(S + 'jev_test.json')) if d.get('ok') and d.get('p') is not None}
R_dev = json.load(open(S + 'dev2000_records.json')); J_dev = {d['id']: d['p'] for d in json.load(open(S + 'jev_dev2000.json')) if d.get('ok') and d.get('p') is not None}
T = per_review(R_test, J_test, 'label', 'test'); T['split'] = 'held-out'
D = per_review(R_dev, J_dev, 'label', 'dev'); D['split'] = 'development'
A = pd.concat([T, D], ignore_index=True); A['year'] = A.review.map(year_of)
bins = [(0, 2019, '2011-2019'), (2020, 2021, '2020-2021'), (2022, 2022, '2022'), (2023, 2023, '2023'), (2024, 2025, '2024-2025')]
A['year_group'] = [next(lbl for lo, hi, lbl in bins if lo <= y <= hi) for y in A.year]
A.to_csv(OUT + '/per_review_year_metrics.csv', index=False)
def strat(df):
    g = df.groupby('year_group')
    t = pd.DataFrame({'n_reviews': g.size(), 'jev_auc_mean': g.jev_auc.mean().round(4), 'bge_auc_mean': g.bge_auc.mean().round(4) if 'bge_auc' in df else np.nan,
                      'tau_reliability_recall_ge95': g.tau_recall.apply(lambda x: (x >= 0.95).mean()).round(4), 'tau_recall_mean': g.tau_recall.mean().round(4), 'tau_recall_min': g.tau_recall.min().round(4), 'tau_work_mean': g.tau_work.mean().round(4)})
    order = [b[2] for b in bins]; t = t.reindex([o for o in order if o in t.index])
    return t
strata = {}
for name, df in [('held-out', T.assign(year=T.review.map(year_of), year_group=A[A.split == 'held-out'].year_group.values)), ('development', D.assign(year=D.review.map(year_of), year_group=A[A.split == 'development'].year_group.values)), ('all_96', A)]:
    t = strat(df); t.to_csv(f'{OUT}/strata_{name}.csv'); strata[name] = json.loads(t.to_json(orient='index'))
    ra, rr = spearmanr(df.year, df.jev_auc), spearmanr(df.year, df.tau_recall)
    rw = spearmanr(df.year, df.tau_work)
    strata[name + '_spearman'] = {'n': len(df), 'year_vs_jev_auc': {'rho': round(float(ra[0]), 4), 'p': round(float(ra[1]), 4)}, 'year_vs_tau_recall': {'rho': round(float(rr[0]), 4), 'p': round(float(rr[1]), 4)}, 'year_vs_tau_work': {'rho': round(float(rw[0]), 4), 'p': round(float(rw[1]), 4)},
                                  'year_min': int(df.year.min()), 'year_max': int(df.year.max()), 'jev_auc_mean': round(float(df.jev_auc.mean()), 4), 'tau_reliability': round(float((df.tau_recall >= 0.95).mean()), 4)}
    if 'bge_auc' in df: 
        rb = spearmanr(df.year, df.jev_auc - df.bge_auc); strata[name + '_spearman']['year_vs_jev_minus_bge_auc'] = {'rho': round(float(rb[0]), 4), 'p': round(float(rb[1]), 4)}
res['year_strata'] = strata
res['year_strata_note'] = 'Year = publication year encoded in the SYNERGY review key. Bins fixed before computation: 2011-2019, 2020-2021, 2022, 2023, 2024-2025. Threshold recall uses tau=0.07 on the batched scores of the original runs. Development set = the 73 development reviews of at most 2000 records that were scored.'
# 2024-2025 reviews listed
res['reviews_2024_2025'] = A[A.year >= 2024][['review', 'split', 'n', 'n1', 'jev_auc', 'tau_recall', 'tau_work']].round(4).to_dict(orient='records')
# label publicity (documented dates)
res['label_publicity'] = {
    'synergy_plus_v3.0_release': {'date_utc': '2026-08-27T08:21:34Z', 'source': 'GitHub API releases for asreview/synergy-dataset (tag synergy_plus_v3.0), fetched 2026-09-26 17:52 EDT; copy in analysis/AN-0001-17/sources/'},
    'synergy_v1.0_release': {'date_utc': '2023-04-24T15:16:39Z', 'source': 'same GitHub API response (tag v1.0)'},
    'clef_2019_qrels': {'year': 2019, 'source': 'CLEF 2019 TAR task overview (manuscript reference 19); qrels distributed with the task'},
    'vendor_model_release': {'date': '2026-09-15', 'source': 'TypeSafe AI blog post (reference 16) and Vercel AI Gateway model listing field released=1789430400 (2026-09-15 00:00 UTC), fetched 2026-09-26'},
    'pilot_review': {'status': 'unpublished; labels produced by the authors group; never public', 'scoring_dates': 'Sept 24, 2026 (screen/ file times)'},
    'per_review_deposit_dates': 'not collected (deposits on OSF, Zenodo, DANS and others; outside this analysis)',
    'consequence': 'Every SYNERGY and CLEF label used was public before the vendor release date; the label-publicity stratum therefore contains all benchmark reviews and only the pilot review lies outside it.'}
# same-label-level table (title and abstract labels)
Tta = per_review(R_test, J_test, 'label_ta', 'test')
# bge at TA level for held-out: scores_zs test files with label_ta
res['ta_heldout'] = {'n_reviews': len(Tta), 'jev_auc_mean': round(float(Tta.jev_auc.mean()), 4), 'bge_auc_mean': round(float(Tta.bge_auc.mean()), 4) if 'bge_auc' in Tta else None, 'minilm_auc_mean': round(float(Tta.minilm_auc.mean()), 4) if 'minilm_auc' in Tta else None,
                     'tau_reliability': round(float((Tta.tau_recall >= 0.95).mean()), 4), 'tau_recall_mean': round(float(Tta.tau_recall.mean()), 4), 'tau_recall_min': round(float(Tta.tau_recall.min()), 4), 'tau_work_mean': round(float(Tta.tau_work.mean()), 4), 'reviews': Tta.review.tolist()}
Tta.to_csv(OUT + '/heldout_ta_per_review.csv', index=False)
p3 = pd.read_csv(S + 'p3_ta_auc.csv'); p3t = p3[p3.split == 'test']; res['ta_heldout']['p3_ta_auc_csv_check'] = {'n': len(p3t), 'jev_ta_mean': round(float(p3t.jev_ta.mean()), 4), 'bge_ta_mean': round(float(p3t.bge_ta.mean()), 4)}
# CLEF at TA level
Rc = json.load(open('/N/project/AiLab/jev/clef/clef_records.json')); Jc = {d['id']: d['p'] for d in json.load(open(S + 'jev_clef.json')) if d.get('ok')}
Cta = per_review(Rc, Jc, 'label_ta', 'clef'); Cfin = per_review(Rc, Jc, 'label', 'clef')
res['ta_clef'] = {'n_reviews': len(Cta), 'jev_auc_mean': round(float(Cta.jev_auc.mean()), 4), 'bge_auc_mean': round(float(Cta.bge_auc.mean()), 4) if 'bge_auc' in Cta else None, 'tau_reliability': round(float((Cta.tau_recall >= 0.95).mean()), 4), 'tau_recall_mean': round(float(Cta.tau_recall.mean()), 4), 'tau_recall_min': round(float(Cta.tau_recall.min()), 4), 'tau_work_mean': round(float(Cta.tau_work.mean()), 4)}
res['final_clef'] = {'n_reviews': len(Cfin), 'jev_auc_mean': round(float(Cfin.jev_auc.mean()), 4), 'bge_auc_mean': round(float(Cfin.bge_auc.mean()), 4) if 'bge_auc' in Cfin else None, 'tau_reliability': round(float((Cfin.tau_recall >= 0.95).mean()), 4), 'tau_work_mean': round(float(Cfin.tau_work.mean()), 4)}
cz = pd.read_csv(S + 'clef_zs_label_ta.csv'); res['ta_clef']['clef_zs_label_ta_csv_check'] = {'n': len(cz), 'jev_AUC_mean': round(float(cz.jev_AUC.mean()), 4), 'bge_AUC_mean': round(float(cz.bge_AUC.mean()), 4)}
Cta.to_csv(OUT + '/clef_ta_per_review.csv', index=False)
# held-out final level (for the table)
res['final_heldout'] = {'n_reviews': len(T), 'jev_auc_mean': round(float(T.jev_auc.mean()), 4), 'bge_auc_mean': round(float(T.bge_auc.mean()), 4), 'tau_reliability': round(float((T.tau_recall >= 0.95).mean()), 4), 'tau_work_mean': round(float(T.tau_work.mean()), 4)}
# pilot review (title and abstract pass labels; records in the PubMed arm)
sc = '/N/project/AiLab/jev/screen/'
sj = [d for d in json.load(open(sc + 'screen_jev.json')) if d.get('ok')]; scl = [d for d in json.load(open(sc + 'screen_claude-opus.json')) if d.get('ok')]
b10 = [d for d in json.load(open(S + 'dhl_jev_b10.json')) if d.get('ok')]
def pilot_stats(D, lab_key='label'):
    y = np.array([d[lab_key] for d in D]); p = np.array([d['p'] for d in D]); s = p >= TAU
    return {'n': len(D), 'n_pos': int(y.sum()), 'auc': round(float(roc_auc_score(y, p)), 4), 'tau_recall': round(float(y[s].sum() / y.sum()), 4), 'tau_work': round(float(s.mean()), 4)}
def pilot_final(D):
    y = np.array([1 if d['stage'] == 'included' else 0 for d in D]); p = np.array([d['p'] for d in D]); s = p >= TAU
    return {'n': len(D), 'n_included_final': int(y.sum()), 'auc_included_vs_rest': round(float(roc_auc_score(y, p)), 4), 'tau_recall_final': round(float(y[s].sum() / y.sum()), 4)}
stage = {d['pmid']: d['stage'] for d in sj}
for d in b10: d['stage'] = stage.get(d['id'])
res['pilot'] = {'jev_single_record_screen_jev': {**pilot_stats(sj), **pilot_final(sj)}, 'jev_batch10_dhl_jev_b10': {**pilot_stats(b10), **pilot_final(b10)},
                'claude_opus_single_record': {**pilot_stats(scl), **pilot_final(scl)},
                'free_rankers_zs_auc_dhl_csv': pd.read_csv(S + 'zs_auc_dhl.csv').iloc[0].to_dict(),
                'label_level': 'title and abstract decision (passed to full text) among 1572 PubMed-arm records; 144 positives; final inclusion 77 of them',
                'note': 'tau=0.07 was selected on batched development scores; applied to single-record pilot scores only for reference.'}
# 544-record four-model subset at the TA level
sub = set(json.load(open(sc + 'subset544.json')))
four = {}
for name, f in [('jev', 'screen_jev.json'), ('claude_opus', 'screen_claude-opus.json'), ('deepseek', 'screen_deepseek.json'), ('gpt4omini_verbalised', 'screen_gpt4omini.json')]:
    D = [d for d in json.load(open(sc + f)) if d.get('ok') and d['pmid'] in sub]; four[name] = {'n': len(D), 'n_pos': int(sum(d['label'] for d in D)), 'auc': round(float(roc_auc_score([d['label'] for d in D], [d['p'] for d in D])), 4), 'distinct_p': len({d['p'] for d in D})}
res['pilot_subset544_four_models'] = four; res['pilot_subset544_note'] = 'GPT-4o-mini in the pilot returned a verbalised probability (screen/run_screen.mjs), not log-probabilities; subset = all 144 positives plus 400 random negatives (seed 7).'
json.dump(res, open(OUT + '/results.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
