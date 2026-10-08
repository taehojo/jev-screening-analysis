#!/usr/bin/env python
# npj AN-0002-02 (manuscript analysis 48, post hoc): which part of each MedCPT Cross-Encoder query-document pair
# was cut when pairs longer than 512 tokens were truncated in analysis 44 (AN-0001-13).
# Pair construction copied from rounds/round_0001/revision/analysis/AN-0001-13/an13_score.py:
#   query = criteria text (synergy/criteria.json, clef/clef_criteria.json), document = f"{title}. {abstract}",
#   pair = [query, document], tokenizer(truncation=False) for the full length, tokenizer(truncation=True, max_length=512).
# Differences from an13_score.py: no model is loaded and nothing is scored; token_type_ids of the truncated
# encoding give the query and document tokens kept; output is written to this folder only.
# Set membership (23 held-out, 28 CLEF content-level, 31 CLEF title-and-abstract reviews) is read from
# AN-0001-13/per_review_auc_tnr95.csv (columns set and review only). No label is read.
import os, json, time, subprocess, csv, gzip
HERE = os.path.dirname(os.path.abspath(__file__))
R1 = '/N/project/AiLab/jev/review_pipeline_npj/rounds/round_0001/revision/analysis/AN-0001-13'
os.environ['HF_HOME'] = os.path.join(R1, 'hf_home'); os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.setdefault('RAYON_NUM_THREADS', '8')
import numpy as np
import transformers, tokenizers
from transformers import AutoTokenizer
ROOT = '/N/project/AiLab/jev'
def now(): return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S %Z'], capture_output=True, text=True).stdout.strip()
D = json.load(open(os.path.join(R1, 'model_download.json'))); path = D['local_path']
tok = AutoTokenizer.from_pretrained(path)
print('tokenizer', D['repo'], D['revision_sha'], type(tok).__name__, 'transformers', transformers.__version__, 'tokenizers', tokenizers.__version__, now(), flush=True)
members = {}
for row in csv.DictReader(open(os.path.join(R1, 'per_review_auc_tnr95.csv'))):
    members.setdefault(row['set'], []).append(row['review'])
SRC = [('test', ROOT + '/synergy/test_records.json', ROOT + '/synergy/criteria.json'), ('clef', ROOT + '/clef/clef_records.json', ROOT + '/clef/clef_criteria.json')]
per_pair = {}   # (src, review) -> arrays
per_review = []
t00 = time.time(); T_START = now()
for name, rp, cp in SRC:
    R = json.load(open(rp)); C = json.load(open(cp)); by = {}
    for r in R: by.setdefault(r['review'], []).append(r)
    for k in sorted(by):
        if k not in C: print('no criteria', k, flush=True); continue
        recs = by[k]; q = C[k]; docs = [f"{r['title']}. {r['abstract']}" for r in recs]
        qf = len(tok(q, add_special_tokens=False)['input_ids'])
        QF, DF, QK, DK, TOT = [], [], [], [], []
        for i in range(0, len(docs), 256):
            pairs = [[q, d] for d in docs[i:i + 256]]
            full = tok(pairs, truncation=False, padding=False)
            tr = tok(pairs, truncation=True, padding=False, max_length=512)
            for fi, ft, ti, tt in zip(full['input_ids'], full['token_type_ids'], tr['input_ids'], tr['token_type_ids']):
                n0f = ft.count(0); n1f = ft.count(1); n0t = tt.count(0); n1t = tt.count(1)
                QF.append(n0f - 2); DF.append(n1f - 1); QK.append(n0t - 2); DK.append(n1t - 1); TOT.append(len(fi))
                assert len(ti) <= 512 and len(ti) == n0t + n1t
        QF, DF, QK, DK, TOT = map(np.array, (QF, DF, QK, DK, TOT))
        assert (QF == qf).all(), (k, qf, QF[:3])
        prev = json.load(open(os.path.join(R1, 'scores', f'{name}_{k}.json')))
        ntr = int((TOT > 512).sum())
        per_pair[(name, k)] = dict(ids=[r['id'] for r in recs], QF=QF, DF=DF, QK=QK, DK=DK, TOT=TOT)
        per_review.append({'src': name, 'review': k, 'pairs': len(docs), 'query_tokens': qf, 'query_tokens_an13': prev['query_tokens'],
                           'truncated': ntr, 'truncated_an13': prev['n_pairs_truncated'],
                           'query_cut': int((QK < QF).sum()), 'doc_cut': int((DK < DF).sum()),
                           'median_doc_tokens': float(np.median(DF))})
        print(f"{name} {k} n={len(docs)} q={qf} trunc={ntr} (an13 {prev['n_pairs_truncated']}) qcut={int((QK < QF).sum())} {time.time()-t00:.0f}s", flush=True)

def summ(setname, src, reviews):
    A = {f: np.concatenate([per_pair[(src, k)][f] for k in reviews]) for f in ['QF', 'DF', 'QK', 'DK', 'TOT']}
    tr = A['TOT'] > 512; qc = A['QK'] < A['QF']; dc = A['DK'] < A['DF']
    rm_doc = (A['DF'] - A['DK'])[tr]; ret_doc = (A['DK'] / np.maximum(A['DF'], 1))[tr]
    rm_q = (A['QF'] - A['QK'])[qc]
    qt = np.array([per_pair[(src, k)]['QF'][0] for k in reviews])
    return {'reviews': len(reviews), 'pairs': int(tr.size), 'pairs_truncated': int(tr.sum()), 'share_truncated': float(tr.mean()),
            'truncated_only_document_shortened': int((tr & dc & ~qc).sum()), 'truncated_only_query_shortened': int((tr & qc & ~dc).sum()),
            'truncated_both_shortened': int((tr & qc & dc).sum()), 'truncated_neither_shortened_check': int((tr & ~qc & ~dc).sum()),
            'not_truncated_but_shortened_check': int((~tr & (qc | dc)).sum()),
            'query_tokens_per_review': {'median': float(np.median(qt)), 'min': int(qt.min()), 'max': int(qt.max())},
            'reviews_with_query_over_509_tokens': int((qt > 509).sum()),
            'reviews_with_query_cut_in_any_pair': int(sum(int((per_pair[(src, k)]['QK'] < per_pair[(src, k)]['QF']).any()) for k in reviews)),
            'document_tokens_all_pairs': {'median': float(np.median(A['DF'])), 'min': int(A['DF'].min()), 'max': int(A['DF'].max())},
            'document_tokens_removed_in_truncated_pairs': {'median': float(np.median(rm_doc)) if rm_doc.size else None, 'min': int(rm_doc.min()) if rm_doc.size else None, 'max': int(rm_doc.max()) if rm_doc.size else None},
            'share_of_document_tokens_retained_in_truncated_pairs': {'median': float(np.median(ret_doc)) if ret_doc.size else None, 'min': float(ret_doc.min()) if ret_doc.size else None, 'max': float(ret_doc.max()) if ret_doc.size else None},
            'share_of_document_tokens_retained_pooled_all_pairs': float(A['DK'].sum() / A['DF'].sum()),
            'query_tokens_removed_where_query_cut': {'n': int(qc.sum()), 'median': float(np.median(rm_q)) if rm_q.size else None, 'max': int(rm_q.max()) if rm_q.size else None},
            'share_of_query_tokens_retained_pooled_all_pairs': float(A['QK'].sum() / A['QF'].sum())}

res = {'analysis': 'AN-0002-02', 'manuscript_analysis_number': 48, 'post_hoc': True, 'started_script': T_START,
       'tokenizer': {'repo': D['repo'], 'revision_sha': D['revision_sha'], 'class': type(tok).__name__,
                     'transformers': transformers.__version__, 'tokenizers': tokenizers.__version__,
                     'truncation_true_maps_to': 'longest_first (transformers/tokenization_utils_base.py: "if truncation is True: truncation_strategy = TruncationStrategy.LONGEST_FIRST")',
                     'tokenizer_config_truncation_strategy': json.load(open(os.path.join(path, 'tokenizer_config.json'))).get('truncation_strategy'),
                     'truncation_side': tok.truncation_side},
       'sets': {}}
res['sets']['heldout_final'] = summ('heldout_final', 'test', members['heldout_final'])
res['sets']['clef_content'] = summ('clef_content', 'clef', members['clef_content'])
res['sets']['clef_ta'] = summ('clef_ta', 'clef', members['clef_ta'])
an13 = json.load(open(os.path.join(R1, 'results.json')))['sets']
res['reproduction_of_an13_counts'] = {s: {'an13': an13[s]['pairs_truncated_at_512'], 'here': res['sets'][s]['pairs_truncated'],
                                          'equal': an13[s]['pairs_truncated_at_512'] == res['sets'][s]['pairs_truncated']} for s in res['sets']}
res['per_review_counts_equal_an13'] = all(r['truncated'] == r['truncated_an13'] and r['query_tokens'] == r['query_tokens_an13'] for r in per_review)
res['finished'] = now()
json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
with open(os.path.join(HERE, 'per_review.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(per_review[0].keys())); w.writeheader(); w.writerows(per_review)
with gzip.open(os.path.join(HERE, 'per_pair_tokens.csv.gz'), 'wt') as f:
    f.write('src,review,id,query_tokens,document_tokens,query_tokens_kept,document_tokens_kept,pair_length_untruncated\n')
    for (src, k), v in per_pair.items():
        for i in range(len(v['ids'])):
            f.write(f"{src},{k},{v['ids'][i]},{v['QF'][i]},{v['DF'][i]},{v['QK'][i]},{v['DK'][i]},{v['TOT'][i]}\n")
print(json.dumps(res['reproduction_of_an13_counts']), 'per-review equal:', res['per_review_counts_equal_an13'], flush=True)
print('done', now(), flush=True)
