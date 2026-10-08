# AN-0002-02 check (run after an48_truncation.py): per-review agreement with AN-0001-13.
# an48_truncation.py counts query tokens without the special tokens ([CLS] and [SEP]); an13_score.py stored
# len(tok(q)['input_ids']), which includes them. This script checks that the two differ by exactly 2 in every review and
# that the per-review numbers of truncated pairs are identical. It reads per_review.csv only and writes check.json.
import csv, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
R = list(csv.DictReader(open(os.path.join(HERE, 'per_review.csv'))))
out = {'reviews': len(R),
       'truncated_equal_in_every_review': all(r['truncated'] == r['truncated_an13'] for r in R),
       'query_tokens_plus_2_equal_an13_in_every_review': all(int(r['query_tokens']) + 2 == int(r['query_tokens_an13']) for r in R),
       'reviews_where_query_cut_in_any_pair': [f"{r['src']}:{r['review']}" for r in R if int(r['query_cut']) > 0]}
json.dump(out, open(os.path.join(HERE, 'check.json'), 'w'), indent=1)
print(json.dumps(out))
