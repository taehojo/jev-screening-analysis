"""AN-0003-02 (round 3, post hoc): did the positive records with a Jev probability below tau = 0.07 share their
ten-record request with another positive record, compared with random order within reviews?

Plan: REVISION_ANALYSIS_LOG.md, entry of 2026-09-28 10:28:33 EDT (AN-0003-02) and the supplement of
2026-09-28 10:34:49 EDT (scores stored to two decimals; identification by public identifiers only).

Collections: held-out SYNERGY+ reviews (final inclusion), CLEF 2019 (content-level label); development reviews
(final inclusion) reported separately. Requests are rebuilt with a copy of the AN-0002-01 reconstruction
(reconstruct_requests) and checked against the consecutive ten-record slices of the data file.
For each selected record: request size n, other positive records in the request, stored score; probability of
sharing under random order within the review with request sizes fixed, 1 - C(N-K, n-1) / C(N-1, n-1).
Totals: observed against expected; exact Poisson-binomial distribution of the count (independence assumed).
Reference: observed and expected share of all positive records of the collection that shared a request.
No record title or abstract is printed or written. No model call; nothing is generated.
"""
import datetime
import json
import math
import os
import platform
import re
import sys
from collections import defaultdict
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/N/project/AiLab/jev"
SYN = ROOT + "/synergy"
APPX = ROOT + "/review_pipeline/versions/v0002_after_round0002/source/appendix_v2.md"
TAU = 0.07
started = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
COLL = [
    ("heldout", SYN + "/test_records.json", SYN + "/jev_test.json"),
    ("clef", ROOT + "/clef/clef_records.json", SYN + "/jev_clef.json"),
    ("development", SYN + "/dev2000_records.json", SYN + "/jev_dev2000.json"),
]


def log(*a):
    print(*a, flush=True)


# ---- copy of AN-0002-01 reconstruct_requests (unchanged logic)
def reconstruct_requests(entries):
    reqs, i = [], 0
    while i < len(entries):
        e = entries[i]
        bsz = int(e.get("batch") or 1)
        seg = [e]
        j = i + 1
        while (j < len(entries) and len(seg) < bsz and entries[j]["review"] == e["review"]
               and entries[j].get("tokens") == e.get("tokens") and entries[j].get("batch") == e.get("batch")):
            seg.append(entries[j]); j += 1
        reqs.append(seg)
        i = j
    return reqs


def p_share(N, K, n):
    """P(at least one of the other n-1 records of the request is positive) under random order within the review."""
    if n <= 1 or K <= 1:
        return Fraction(0)
    return 1 - Fraction(math.comb(N - K, n - 1), math.comb(N - 1, n - 1))


def poisson_binomial(ps):
    dist = np.array([1.0])
    for p in ps:
        dist = np.convolve(dist, [1 - p, p])
    return dist


def public_id(coll, rid):
    parts = rid.split("|")
    if coll == "clef":
        return {"pmid": parts[1]}
    oa = parts[1].rsplit("/", 1)[-1] if len(parts) > 1 else None
    return {"openalex": oa, "source_row": int(parts[-1]) if len(parts) > 2 else None}


def appendix_tables():
    """Rows of tables S5 and S6 of the v0002 appendix (review, split, stored probability; title kept in memory only)."""
    txt = open(APPX, encoding="utf-8").read().splitlines()
    s5, s6 = [], []
    mode = None
    for line in txt:
        if line.startswith("**Table S5."):
            mode = "s5"; continue
        if line.startswith("**Table S6."):
            mode = "s6"; continue
        if line.startswith("### ") or line.startswith("## "):
            mode = None
        if mode and line.startswith("| ") and not line.startswith("| Review") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if mode == "s5":
                s5.append({"review": cells[0].replace(" ", "_"), "split": cells[1], "p": float(cells[2].replace("·", ".")), "title": cells[3]})
            else:
                m = re.search(r"PMID (\d+).*\((\d)·(\d+)\)", cells[2])
                rv = cells[0].split(" ")[0]
                s6.append({"review": rv, "pmid": m.group(1), "p": float(f"{m.group(2)}.{m.group(3)}")})
    return s5, s6


def main():
    out = {"analysis_id": "AN-0003-02", "post_hoc": True, "tau": TAU, "started": started,
           "environment": {"python": platform.python_version(), "numpy": np.__version__, "executable": sys.executable},
           "definitions": {"selected": "positive records (final inclusion; CLEF content-level label) with stored Jev probability < 0.07",
                           "p_share": "1 - C(N-K, n-1)/C(N-1, n-1); N records and K positive records of the review, n size of the record's request",
                           "poisson_binomial": "exact distribution of the number of selected records sharing a request, assuming independence between records",
                           "threshold_rule": "records with p >= tau are read (synergy/stop_eval.py, final_eval.py)",
                           "score_precision": "stored Jev probabilities have at most two decimals"},
           "collections": {}}
    s5, s6 = appendix_tables()
    for coll, rf, jf in COLL:
        recs = json.load(open(rf))
        jev = json.load(open(jf))
        lab = {r["id"]: (None if r.get("label") in (None, "None") else int(r["label"])) for r in recs}
        order = defaultdict(list)
        for r in recs:
            order[r["review"]].append(r["id"])
        reqs = reconstruct_requests(jev)
        det = set()
        for k, ids in order.items():
            for i in range(0, len(ids), 10):
                det.add(frozenset(ids[i:i + 10]))
        n_equal = sum(1 for s in reqs if frozenset(e["id"] for e in s) in det)
        req_of = {}
        for gi, s in enumerate(reqs):
            for e in s:
                req_of[e["id"]] = gi
        score = {e["id"]: float(e["p"]) for e in jev if e.get("p") is not None}
        NK = {}
        for k, ids in order.items():
            ys = [lab[i] for i in ids if lab[i] is not None]
            NK[k] = (len(ys), sum(ys))
        # all positive records: observed and expected sharing
        all_obs, all_exp, n_pos = 0, Fraction(0), 0
        sel = []
        for k, ids in order.items():
            N, K = NK[k]
            for i in ids:
                if lab[i] != 1:
                    continue
                g = reqs[req_of[i]]
                n = sum(1 for e in g if lab[e["id"]] is not None)
                others = sum(1 for e in g if e["id"] != i and lab[e["id"]] == 1)
                ps = p_share(N, K, n)
                n_pos += 1
                all_obs += int(others > 0)
                all_exp += ps
                if score[i] < TAU:
                    oth_below = sum(1 for e in g if e["id"] != i and lab[e["id"]] == 1 and score[e["id"]] < TAU)
                    d = {"review": k, **public_id(coll, i), "stored_score": score[i], "request_size": n,
                         "other_positive_in_request": others, "other_positive_below_tau": oth_below,
                         "other_positive_at_or_above_tau": others - oth_below, "request_index": req_of[i],
                         "shared": others > 0, "p_share_random": float(ps),
                         "review_N": N, "review_K": K, "position_in_file": ids.index(i)}
                    sel.append(d)
        ps = [d["p_share_random"] for d in sel]
        obs = sum(int(d["shared"]) for d in sel)
        dist = poisson_binomial(ps)
        C = {"records_file": rf, "jev_file": jf, "requests": len(reqs), "requests_equal_to_file_slices": n_equal,
             "n_selected": len(sel), "observed_shared": obs, "expected_shared": float(sum(ps)),
             "P_X_ge_observed": float(dist[obs:].sum()), "P_X_le_observed": float(dist[:obs + 1].sum()),
             "poisson_binomial_pmf": [float(x) for x in dist],
             "reference_all_positive": {"n_positive": n_pos, "observed_shared": all_obs, "observed_share": all_obs / n_pos,
                                        "expected_shared": float(all_exp), "expected_share": float(all_exp / n_pos)},
             "selected": sel}
        # identity check against the appendix tables (titles compared in memory, not written)
        if coll in ("heldout", "development"):
            split = "test" if coll == "heldout" else "dev"
            ref = [r for r in s5 if r["split"] == split]
            title = {r["id"]: r["title"] for r in recs}
            sel_ids = [r["id"] for r in recs if lab[r["id"]] == 1 and score[r["id"]] < TAU]
            matched = 0
            unmatched_rows = []
            used = set()
            for t in ref:
                hit = [i for i in sel_ids if i not in used and i.split("|")[0] == t["review"] and abs(score[i] - t["p"]) < 1e-9
                       and title[i].strip().startswith(t["title"].strip()[:60])]
                if len(hit) == 1:
                    matched += 1; used.add(hit[0])
                else:
                    unmatched_rows.append({"review": t["review"], "p": t["p"], "n_candidates": len(hit)})
            C["appendix_check"] = {"table": "S5", "rows_in_table": len(ref), "selected_here": len(sel_ids), "rows_matched_one_to_one": matched,
                                   "unmatched_table_rows": unmatched_rows, "selected_not_in_table": len(sel_ids) - len(used)}
        else:
            ref = s6
            sel_keys = {(d["review"], d["pmid"]): d["stored_score"] for d in sel}
            matched = sum(1 for t in ref if (t["review"], t["pmid"]) in sel_keys and abs(sel_keys[(t["review"], t["pmid"])] - t["p"]) < 1e-9)
            C["appendix_check"] = {"table": "S6", "rows_in_table": len(ref), "selected_here": len(sel), "rows_matched": matched,
                                   "selected_not_in_table": [{"review": k[0], "pmid": k[1]} for k in sel_keys if not any((t["review"], t["pmid"]) == k for t in ref)]}
        out["collections"][coll] = C
        log(f"== {coll}: requests {len(reqs)}, equal to file slices {n_equal}; selected {len(sel)}; shared {obs} vs expected {C['expected_shared']:.3f}; "
            f"P(X>=obs)={C['P_X_ge_observed']:.4f} P(X<=obs)={C['P_X_le_observed']:.4f}")
        log(f"   reference all positives: {C['reference_all_positive']}")
        log(f"   appendix check: {C['appendix_check']}")
        for d in sel:
            log(f"   {d['review']} {({k: d[k] for k in ('openalex', 'pmid') if k in d})} score {d['stored_score']:.2f} n={d['request_size']} other_pos={d['other_positive_in_request']} "
                f"(below tau {d['other_positive_below_tau']}, at or above {d['other_positive_at_or_above_tau']}; request {d['request_index']}) "
                f"p_share={d['p_share_random']:.4f} (N={d['review_N']}, K={d['review_K']})")
    # evaluation sets pooled (held-out + CLEF), as requested by the comment; development not pooled
    ev = out["collections"]["heldout"]["selected"] + out["collections"]["clef"]["selected"]
    ps = [d["p_share_random"] for d in ev]
    obs = sum(int(d["shared"]) for d in ev)
    dist = poisson_binomial(ps)
    ra = [out["collections"][c]["reference_all_positive"] for c in ("heldout", "clef")]
    out["evaluation_sets_pooled"] = {"n_selected": len(ev), "observed_shared": obs, "expected_shared": float(sum(ps)),
                                     "P_X_ge_observed": float(dist[obs:].sum()), "P_X_le_observed": float(dist[:obs + 1].sum()),
                                     "poisson_binomial_pmf": [float(x) for x in dist],
                                     "reference_all_positive": {"n_positive": sum(r["n_positive"] for r in ra), "observed_shared": sum(r["observed_shared"] for r in ra),
                                                                "expected_shared": sum(r["expected_shared"] for r in ra)}}
    rp = out["evaluation_sets_pooled"]["reference_all_positive"]
    rp["observed_share"] = rp["observed_shared"] / rp["n_positive"]; rp["expected_share"] = rp["expected_shared"] / rp["n_positive"]
    log(f"== evaluation sets pooled: selected {len(ev)}, shared {obs} vs expected {sum(ps):.3f}; P(X>=obs)={dist[obs:].sum():.4f} P(X<=obs)={dist[:obs + 1].sum():.4f}")
    log(f"   reference: {rp}")
    # Kennis 2004 (CD011977, PMID 15682919)
    k = [d for d in out["collections"]["clef"]["selected"] if d["review"] == "CD011977" and d.get("pmid") == "15682919"]
    if k:
        s = k[0]["stored_score"]
        out["kennis_2004"] = {"review": "CD011977", "pmid": "15682919", "stored_score": s, "stored_score_repr": repr(s),
                              "at_least_0.03": bool(s >= 0.03), "distance_from_0.03": s - 0.03,
                              "read_under_tau_0.03_rule_p_ge_tau": bool(s >= 0.03), "read_under_tau_0.07": bool(s >= 0.07),
                              "note": "stored to two decimals; the distance is known only at that precision"}
    else:
        out["kennis_2004"] = {"found_among_selected": False}
    log(f"== Kennis 2004: {out['kennis_2004']}")
    out["finished"] = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float)
    log("finished", out["finished"])


if __name__ == "__main__":
    main()
