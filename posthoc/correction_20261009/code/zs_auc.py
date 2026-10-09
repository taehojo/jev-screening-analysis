"""AUCs of the seven free rankers on the pilot review, as in synergy/zs_baselines.py (roc_auc_score of the
title and abstract label against each stored score), with the original and the corrected labels."""
import json
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = "/N/project/AiLab/jev"
S = json.load(open(f"{ROOT}/synergy/scores_zs/dhl_dhl.json"))
rows = []
for tag, path in [("original", f"{ROOT}/synergy/dhl_records.json"), ("corrected", f"{ROOT}/relabel_20261009/data/dhl_records.json")]:
    R = json.load(open(path))
    y = [int(r["label"]) for r in R]
    row = {"review": "dhl", "n": len(R), "incl": sum(y)}
    for m in ["tfidf", "bm25", "minilm", "bge", "e5", "ce_bge", "ce_m3"]:
        assert len(S[m]) == len(R)
        row[m] = roc_auc_score(y, S[m])
    rows.append((tag, row))
stored = pd.read_csv(f"{ROOT}/synergy/zs_auc_dhl.csv").iloc[0]
orig = rows[0][1]
for m in ["tfidf", "bm25", "minilm", "bge", "e5", "ce_bge", "ce_m3"]:
    assert abs(orig[m] - stored[m]) < 1e-12, (m, orig[m], stored[m])
print("original labels reproduce zs_auc_dhl.csv")
pd.DataFrame([rows[1][1]]).to_csv(sys.argv[1], index=False)
for tag, r in rows:
    print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
