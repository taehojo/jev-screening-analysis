"""AN-0003-01 (round 3, post hoc): direction of the file-position differences and the effect of data-file order
among tied Jev scores.

Plan: REVISION_ANALYSIS_LOG.md, entry of 2026-09-28 10:28:33 EDT (AN-0003-01) and the supplement of
2026-09-28 10:34:49 EDT (pilot rows filtered; development recomputed; definitions).

(A) Direction of the file-position difference for the reviews with p < 0.05 in AN-0002-01 (final inclusion; CLEF
    content-level label), from the stored per-review values; AUC recomputed with a copy of the AN-0002-01 function.
(B) Tie groups (identical stored Jev probability within a review): mixed groups, included records in them, and the
    within-tie concordance of data-file order (share of (included, excluded) pairs in the same tie group in which the
    included record comes first in the data file).
(C) Per-review WSS@95 with ties in file order minus the mean over ten random-tie seeds (0 to 9); stored values of
    AN-0001-02 for held-out and CLEF; development recomputed with copies of the AN-0001-02 ranking functions; the
    stored collections are recomputed as a check.
Descriptive only. No model call. No data are generated; the random keys only break ties among observed scores, as in
AN-0001-02. Pilot review rows of the stored AN-0001-02 files are dropped on reading and never printed.
"""
import csv
import datetime
import json
import math
import os
import platform
import sys
from collections import defaultdict

import numpy as np
import scipy
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from al_sim_copy import metrics  # unchanged copy of synergy/al_sim.py (md5 38a695d0a7711a37ecca3d32310d1fcc)

ROOT = "/N/project/AiLab/jev"
SYN = ROOT + "/synergy"
RP = ROOT + "/review_pipeline/rounds"
AN0201 = RP + "/round_0002/revision/analysis/AN-0002-01/results.json"
AN0102 = RP + "/round_0001/revision/analysis/AN-0001-02"
SEEDS = list(range(10))
started = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def log(*a):
    print(*a, flush=True)


# ---- copies of AN-0001-02 functions (unchanged logic)
def order_random(p, seed):
    rng = np.random.default_rng(seed)
    N = len(p)
    return np.lexsort((rng.random(N), -p))


def order_stable(p):
    return np.argsort(-p, kind="stable")


def load_by(path):
    by = {}
    for r in json.load(open(path)):
        by.setdefault(r["review"], []).append(r)
    return by


def load_jev(path):
    return {d["id"]: d["p"] for d in json.load(open(path)) if d.get("ok") and d.get("p") is not None}


# ---- copy of AN-0002-01 position_auc (unchanged logic)
def lab(v):
    if v is None or v == "None" or (isinstance(v, float) and math.isnan(v)):
        return None
    return int(v)


def position_auc_one(ys):
    pos = np.arange(len(ys))
    y = np.array([-1 if v is None else v for v in ys])
    a, b = pos[y == 1], pos[y == 0]
    mw = mannwhitneyu(a, b, alternative="two-sided", method="auto")
    return {"n1": int(len(a)), "n0": int(len(b)), "auc_position": float(mw.statistic / (len(a) * len(b))), "p": float(mw.pvalue)}


# ---- collections as in AN-0001-02 (records file order = data-file order within review)
def collection_defs():
    ta_revs = sorted({d["review"] for d in json.load(open(SYN + "/asr_test_ta.json"))})
    clef_revs = sorted({d["review"] for d in json.load(open(SYN + "/asr_clef.json"))})
    return [
        # name, records, jev, label key, review restriction, AN-0001-02 stored collection, AN-0002-01 collection
        ("development_final", SYN + "/dev2000_records.json", SYN + "/jev_dev2000.json", "label", None, None, "development"),
        ("heldout_final", SYN + "/test_records.json", SYN + "/jev_test.json", "label", None, "heldout_final", "heldout"),
        ("heldout_ta", SYN + "/test_records.json", SYN + "/jev_test.json", "label_ta", ta_revs, "heldout_ta", None),
        ("clef_final", ROOT + "/clef/clef_records.json", SYN + "/jev_clef.json", "label", clef_revs, "clef_final", "clef"),
        ("clef_ta", ROOT + "/clef/clef_records.json", SYN + "/jev_clef.json", "label_ta", None, "clef_ta_jevonly_only", None),
    ]


def tie_stats(p, y):
    """Tie groups by identical stored probability; positions are data-file indices within the review."""
    groups = defaultdict(list)
    for i, v in enumerate(p):
        groups[float(v)].append(i)
    mixed = 0
    inc_in_mixed = 0
    pairs = 0
    conc = 0
    for v, idx in groups.items():
        inc = [i for i in idx if y[i] == 1]
        exc = [i for i in idx if y[i] == 0]
        if inc and exc:
            mixed += 1
            inc_in_mixed += len(inc)
            exc_sorted = np.sort(np.array(exc))
            for i in inc:
                # excluded records later in the file than this included record
                conc += int(len(exc_sorted) - np.searchsorted(exc_sorted, i, side="right"))
            pairs += len(inc) * len(exc)
    return {"n_tie_groups": len(groups), "n_mixed_tie_groups": mixed, "included_in_mixed_groups": inc_in_mixed,
            "tie_pairs": pairs, "tie_pairs_included_first": conc,
            "within_tie_concordance": (conc / pairs) if pairs else None}


def read_stored():
    """Stored AN-0001-02 per-review values; pilot rows dropped."""
    per = {}
    with open(AN0102 + "/per_review.csv") as f:
        for r in csv.DictReader(f):
            if r["collection"] == "pilot_review":
                continue
            per[(r["collection"], r["review"])] = r
    seeds = defaultdict(dict)
    with open(AN0102 + "/jev_only_per_seed.csv") as f:
        for r in csv.DictReader(f):
            if r["collection"] == "pilot_review":
                continue
            seeds[(r["collection"], r["review"])][int(r["seed"])] = float(r["wss95"])
    return per, seeds


def main():
    out = {"analysis_id": "AN-0003-01", "post_hoc": True, "started": started,
           "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "executable": sys.executable},
           "inputs": {"an0002_01_results": AN0201, "an0001_02_per_review": AN0102 + "/per_review.csv",
                      "an0001_02_per_seed": AN0102 + "/jev_only_per_seed.csv"},
           "definitions": {
               "direction": "later if auc_position > 0.5 (included records later in the data file than excluded records), earlier if < 0.5",
               "tie_group": "records of one review with identical stored Jev probability (stored to two decimals)",
               "within_tie_concordance": "share of (included, excluded) pairs in the same tie group in which the included record comes first in the data file; 0.5 = no advantage",
               "wss95_diff": "WSS@95 with ties in data-file order (stable sort) minus mean WSS@95 over random tie-break seeds 0-9",
               "contribution": "wss95_diff / number of reviews of the collection (sums to the macro difference)"},
           "collections": {}}
    A = json.load(open(AN0201))
    stored_per, stored_seeds = read_stored()
    rows = []
    for name, rf, jf, lk, restrict, stored_name, an0201_name in collection_defs():
        by = load_by(rf)
        J = load_jev(jf)
        revs = [k for k in by if all(r["id"] in J for r in by[k])]
        if restrict is not None:
            revs = [k for k in revs if k in set(restrict)]
        per_rev = {}
        for k in revs:
            rs = by[k]
            if any(r.get(lk) is None for r in rs):
                continue
            y = np.array([int(r[lk]) for r in rs])
            N = len(y)
            n1 = int(y.sum())
            if not (0 < n1 < N):
                continue
            p = np.array([float(J[r["id"]]) for r in rs])
            st = metrics(order_stable(p), y)["wss95"]
            rnd = [metrics(order_random(p, s), y)["wss95"] for s in SEEDS]
            d = {"collection": name, "review": k, "N": N, "n1": n1,
                 "wss95_file_order_recomputed": float(st), "wss95_random_mean_recomputed": float(np.mean(rnd))}
            d.update(tie_stats(p, y))
            if stored_name is not None:
                s = stored_per.get((stored_name, k))
                sd = stored_seeds.get((stored_name, k), {})
                d["wss95_file_order_stored"] = float(s["jevonly_stable_wss95"]) if s else None
                d["wss95_random_mean_stored"] = float(s["jevonly_wss95"]) if s else None
                d["stored_seed_values_match_recomputed"] = bool(sd and all(abs(sd[i] - metrics(order_random(p, i), y)["wss95"]) < 1e-12 for i in SEEDS))
            if stored_name is not None and d["wss95_file_order_stored"] is not None:
                d["wss95_file_order"] = d["wss95_file_order_stored"]
                d["wss95_random_mean"] = d["wss95_random_mean_stored"]
                d["value_source"] = "stored AN-0001-02"
            else:
                d["wss95_file_order"] = d["wss95_file_order_recomputed"]
                d["wss95_random_mean"] = d["wss95_random_mean_recomputed"]
                d["value_source"] = "recomputed (copy of AN-0001-02 functions)"
            d["wss95_diff"] = d["wss95_file_order"] - d["wss95_random_mean"]
            per_rev[k] = d
        n = len(per_rev)
        for d in per_rev.values():
            d["contribution_to_macro_diff"] = d["wss95_diff"] / n
        # (A) direction, final label level only (AN-0002-01 stored values)
        dir_rows = {}
        if an0201_name is not None:
            pa = A["collections"][an0201_name]["position_auc"]["label"]["per_review"]
            for k, v in pa.items():
                if v["p"] < 0.05:
                    rec = position_auc_one([lab(r["label"]) for r in by[k]])
                    dir_rows[k] = {"review": k, "n1": v["n1"], "n0": v["n0"], "auc_position": v["auc_position"], "p": v["p"],
                                   "direction": "later" if v["auc_position"] > 0.5 else ("earlier" if v["auc_position"] < 0.5 else "none"),
                                   "auc_recomputed": rec["auc_position"], "p_recomputed": rec["p"],
                                   "recomputed_equal": abs(rec["auc_position"] - v["auc_position"]) < 1e-12 and abs(rec["p"] - v["p"]) < 1e-12 and rec["n1"] == v["n1"] and rec["n0"] == v["n0"]}
                    if k in per_rev:
                        dir_rows[k].update({x: per_rev[k][x] for x in ["wss95_diff", "contribution_to_macro_diff", "within_tie_concordance",
                                                                       "n_mixed_tie_groups", "included_in_mixed_groups", "tie_pairs"]})
                        per_rev[k]["direction_p_lt_0.05"] = dir_rows[k]["direction"]
                        per_rev[k]["auc_position"] = v["auc_position"]
                        per_rev[k]["p_position"] = v["p"]
                    else:
                        dir_rows[k]["note"] = "review not in the WSS@95 set of this collection"
        # summaries
        V = list(per_rev.values())
        conc = [d["within_tie_concordance"] for d in V if d["within_tie_concordance"] is not None]
        pairs = sum(d["tie_pairs"] for d in V)
        pc = sum(d["tie_pairs_included_first"] for d in V)
        diffs = np.array([d["wss95_diff"] for d in V])
        macro_file = float(np.mean([d["wss95_file_order"] for d in V]))
        macro_rand = float(np.mean([d["wss95_random_mean"] for d in V]))

        def cls_c(c):
            return "none" if c is None else ("gt_0.5" if c > 0.5 else ("lt_0.5" if c < 0.5 else "eq_0.5"))

        def cls_d(x):
            return "positive" if x > 1e-12 else ("negative" if x < -1e-12 else "zero")

        xtab = defaultdict(int)
        for d in V:
            xtab[f"diff_{cls_d(d['wss95_diff'])}|conc_{cls_c(d['within_tie_concordance'])}"] += 1
        dir_sum = defaultdict(float)
        dir_n = defaultdict(int)
        for d in V:
            g = d.get("direction_p_lt_0.05", "p_ge_0.05")
            dir_sum[g] += d["contribution_to_macro_diff"]
            dir_n[g] += 1
        rec_check = [d for d in V if "wss95_file_order_stored" in d and d["wss95_file_order_stored"] is not None]
        C = {"n_reviews": n, "label": lk, "value_source": V[0]["value_source"] if V else None,
             "macro_wss95_file_order": macro_file, "macro_wss95_random_mean": macro_rand, "macro_diff": macro_file - macro_rand,
             "sum_of_contributions": float(sum(d["contribution_to_macro_diff"] for d in V)),
             "per_review_diff": {"n_positive": int((diffs > 1e-12).sum()), "n_negative": int((diffs < -1e-12).sum()), "n_zero": int((np.abs(diffs) <= 1e-12).sum()),
                                 "min": float(diffs.min()), "max": float(diffs.max()), "median": float(np.median(diffs))},
             "ties": {"reviews_with_mixed_tie_group": sum(1 for d in V if d["n_mixed_tie_groups"] > 0),
                      "mixed_tie_groups": int(sum(d["n_mixed_tie_groups"] for d in V)),
                      "included_records": int(sum(d["n1"] for d in V)),
                      "included_in_mixed_groups": int(sum(d["included_in_mixed_groups"] for d in V)),
                      "pairs": int(pairs), "pairs_included_first": int(pc),
                      "pooled_concordance": (pc / pairs) if pairs else None,
                      "per_review_concordance": {"n": len(conc), "median": float(np.median(conc)) if conc else None,
                                                 "min": float(min(conc)) if conc else None, "max": float(max(conc)) if conc else None,
                                                 "n_gt_0.5": sum(1 for c in conc if c > 0.5), "n_lt_0.5": sum(1 for c in conc if c < 0.5),
                                                 "n_eq_0.5": sum(1 for c in conc if c == 0.5)}},
             "crosstab_diff_by_concordance": dict(xtab),
             "contribution_by_direction_group": {g: {"n_reviews": dir_n[g], "sum_contribution": dir_sum[g]} for g in dir_sum},
             "reproduction_check": {"n_reviews_with_stored_values": len(rec_check),
                                    "file_order_equal": sum(1 for d in rec_check if abs(d["wss95_file_order_stored"] - d["wss95_file_order_recomputed"]) < 1e-12),
                                    "random_mean_equal": sum(1 for d in rec_check if abs(d["wss95_random_mean_stored"] - d["wss95_random_mean_recomputed"]) < 1e-12),
                                    "seed_values_equal": sum(1 for d in rec_check if d.get("stored_seed_values_match_recomputed"))},
             "direction_p_lt_0.05": dir_rows,
             "per_review": per_rev}
        out["collections"][name] = C
        rows.extend(V)
        log(f"== {name}: n={n} ({C['value_source']}); macro file {macro_file:.4f} random {macro_rand:.4f} diff {C['macro_diff']:+.5f}")
        log(f"   ties: mixed groups {C['ties']['mixed_tie_groups']} in {C['ties']['reviews_with_mixed_tie_group']} reviews; included in mixed {C['ties']['included_in_mixed_groups']} of {C['ties']['included_records']}; pooled concordance {C['ties']['pooled_concordance']}")
        log(f"   per-review concordance {C['ties']['per_review_concordance']}")
        log(f"   per-review diff {C['per_review_diff']}")
        log(f"   crosstab {dict(xtab)}")
        log(f"   contribution by direction group {C['contribution_by_direction_group']}")
        log(f"   reproduction check {C['reproduction_check']}")
        for k, v in dir_rows.items():
            log(f"   p<0.05: {k} n1={v['n1']} n0={v['n0']} AUC={v['auc_position']:.3f} p={v['p']:.4f} {v['direction']} recomputed_equal={v['recomputed_equal']}"
                + (f" diff={v['wss95_diff']:+.5f} contrib={v['contribution_to_macro_diff']:+.5f} conc={v['within_tie_concordance']}" if 'wss95_diff' in v else f" {v.get('note', '')}"))
    out["finished"] = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float)
    cols = ["collection", "review", "N", "n1", "value_source", "wss95_file_order", "wss95_random_mean", "wss95_diff", "contribution_to_macro_diff",
            "n_tie_groups", "n_mixed_tie_groups", "included_in_mixed_groups", "tie_pairs", "tie_pairs_included_first", "within_tie_concordance",
            "direction_p_lt_0.05", "auc_position", "p_position", "wss95_file_order_recomputed", "wss95_random_mean_recomputed"]
    with open(os.path.join(HERE, "per_review_ties.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for d in rows:
            w.writerow(d)
    log("finished", out["finished"])


if __name__ == "__main__":
    main()
