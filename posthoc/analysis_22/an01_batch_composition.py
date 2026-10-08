"""AN-0002-01 (round 2, post hoc): record order in the data files and clustering of included records in the
ten-record Jev requests.

Plan: REVISION_ANALYSIS_LOG.md, entries of 2026-09-27 20:48:01 EDT and the AN-0002-01 change entry of the
ANALYZE stage (request membership reconstructed from the output files, since the `batch` field holds the request
size and not a request number).

Inputs (read only): synergy/{dev2000_records,test_records}.json, clef/clef_records.json,
synergy/{jev_dev2000,jev_test,jev_clef}.json, synergy/data/<review>.csv (source order of SYNERGY),
clef/tar/2019-TAR/Task2/Testing/*/qrels/*.content.* (CLEF order).

(A) File order. Mechanism checks and, per review, the AUC of file position for included versus excluded records
    (Mann-Whitney U / (n1 n0); scipy.stats.mannwhitneyu two-sided, method 'auto').
(B) Batch composition. Primary: number of requests with >= 2 included records; expectation under random order
    within review with request sizes fixed, sum over requests of P(X >= 2), X ~ hypergeometric(N, K, n_b).
    Secondary: number of within-request pairs of included records, expectation sum C(n_b, 2) K (K - 1) / (N (N - 1)).
    Permutation test: 10 000 permutations of the observed labels within each review (seed 20260927; a fresh
    numpy default_rng(20260927) per collection and label level), one-sided p for excess and two-sided p.
(C) Descriptive, confounded: within-review difference in mean Jev score of included records that shared their
    request with another included record minus those that did not, averaged over reviews with both kinds.
No model call. No data are generated; the permutation reorders observed labels.
"""
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, mannwhitneyu

ROOT = "/N/project/AiLab/jev"
OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 20260927
B = 10000
CHUNK = 500
COLL = {
    "development": (f"{ROOT}/synergy/dev2000_records.json", f"{ROOT}/synergy/jev_dev2000.json"),
    "heldout": (f"{ROOT}/synergy/test_records.json", f"{ROOT}/synergy/jev_test.json"),
    "clef": (f"{ROOT}/clef/clef_records.json", f"{ROOT}/synergy/jev_clef.json"),
}


def lab(v):
    if v is None or v == "None" or (isinstance(v, float) and math.isnan(v)):
        return None
    return int(v)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ requests
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


# ------------------------------------------------------------------ (A) order mechanism
def order_checks(coll, recs):
    by = defaultdict(list)
    for r in recs:
        by[r["review"]].append(r)
    out = {"n_reviews": len(by)}
    if coll in ("development", "heldout"):
        inc, contig, sorted_oa, sorted_oa_num = 0, 0, 0, 0
        details = {}
        for k, rs in by.items():
            idx = [int(r["id"].rsplit("|", 1)[1]) for r in rs]
            inc += all(a < b for a, b in zip(idx, idx[1:]))
            contig += idx == list(range(len(idx)))
            d = pd.read_csv(f"{ROOT}/synergy/data/{k}.csv", usecols=["openalex_id"])
            oa = d.openalex_id.astype(str).tolist()
            s_lex = oa == sorted(oa)
            nums = [int(re.sub(r"\D", "", x) or -1) for x in oa]
            s_num = nums == sorted(nums)
            sorted_oa += s_lex; sorted_oa_num += s_num
            details[k] = {"n_file": len(rs), "n_source_csv": len(d), "row_index_increasing": all(a < b for a, b in zip(idx, idx[1:])),
                          "all_source_rows_in_order": idx == list(range(len(idx))), "source_sorted_by_openalex_id_text": s_lex,
                          "source_sorted_by_openalex_id_number": s_num}
        out.update({"reviews_row_index_increasing": inc, "reviews_all_source_rows_0_to_n_minus_1": contig,
                    "reviews_source_sorted_by_openalex_id_text": sorted_oa, "reviews_source_sorted_by_openalex_id_number": sorted_oa_num,
                    "per_review": details})
    else:
        qorder = {}
        import glob
        for f in glob.glob(f"{ROOT}/clef/tar/2019-TAR/Task2/Testing/*/qrels/*.content.*"):
            for line in open(f):
                p = line.split()
                if len(p) >= 4:
                    qorder.setdefault(p[0], []).append(p[2])
        same, s_lex, s_num = 0, 0, 0
        details = {}
        for k, rs in by.items():
            pm = [r["id"].split("|")[1] for r in rs]
            q = [x for x in qorder[k] if x in set(pm)]
            eq = pm == q
            sl = pm == sorted(pm); sn = [int(x) for x in pm] == sorted(int(x) for x in pm)
            same += eq; s_lex += sl; s_num += sn
            details[k] = {"n_file": len(rs), "n_qrels": len(qorder[k]), "order_equals_qrels_line_order": eq,
                          "sorted_by_pmid_text": sl, "sorted_by_pmid_number": sn}
        out.update({"reviews_order_equals_qrels_line_order": same, "reviews_sorted_by_pmid_text": s_lex,
                    "reviews_sorted_by_pmid_number": s_num, "per_review": details})
    return out


def position_auc(recs, level):
    by = defaultdict(list)
    for r in recs:
        by[r["review"]].append(lab(r[level]))
    res = {}
    for k, ys in by.items():
        pos = np.arange(len(ys))
        y = np.array([-1 if v is None else v for v in ys])
        a, b = pos[y == 1], pos[y == 0]
        if len(a) == 0 or len(b) == 0:
            continue
        mw = mannwhitneyu(a, b, alternative="two-sided", method="auto")
        res[k] = {"n1": int(len(a)), "n0": int(len(b)), "auc_position": float(mw.statistic / (len(a) * len(b))), "p": float(mw.pvalue)}
    return res


# ------------------------------------------------------------------ (B) batch composition
def review_structs(recs, reqs_by_review, level):
    """For each review: list of request index arrays over labelled records, labels, K, N, request sizes."""
    lab_by_id = {r["id"]: lab(r[level]) for r in recs}
    S = {}
    for k, reqs in reqs_by_review.items():
        ids, gid = [], []
        for g, seg in enumerate(reqs):
            for e in seg:
                if lab_by_id[e["id"]] is not None:
                    ids.append(e["id"]); gid.append(g)
        if not ids:
            continue
        y = np.array([lab_by_id[i] for i in ids], dtype=np.int64)
        gid = np.array(gid)
        sizes = np.bincount(gid, minlength=len(reqs))
        S[k] = {"y": y, "gid": gid, "sizes": sizes, "N": int(len(y)), "K": int(y.sum()), "ids": ids}
    return S


def stats_for(y, gid, nreq):
    cnt = np.bincount(gid, weights=y, minlength=nreq).astype(np.int64)
    return int((cnt >= 2).sum()), int((cnt * (cnt - 1) // 2).sum())


def expected(N, K, sizes):
    e2 = sum(float(hypergeom.sf(1, N, K, n)) for n in sizes if n > 0)
    ep = sum(n * (n - 1) / 2 for n in sizes) * (K * (K - 1)) / (N * (N - 1)) if N > 1 else 0.0
    return e2, ep


def permute(S, rng):
    """Collection totals of the two statistics under 10 000 within-review permutations."""
    tot2 = np.zeros(B, dtype=np.int64); totp = np.zeros(B, dtype=np.int64)
    for k in S:  # insertion order of reviews in the data file
        s = S[k]; N, K = s["N"], s["K"]
        if K < 2:
            continue
        nreq = len(s["sizes"]); gid = s["gid"]
        for c0 in range(0, B, CHUNK):
            c = min(CHUNK, B - c0)
            keys = rng.random((c, N))
            pos = np.argpartition(keys, K - 1, axis=1)[:, :K] if K < N else np.tile(np.arange(N), (c, 1))
            g = gid[pos]  # (c, K) request index of each permuted positive
            flat = (g + (np.arange(c)[:, None] * nreq)).ravel()
            cnt = np.bincount(flat, minlength=c * nreq).reshape(c, nreq)
            tot2[c0:c0 + c] += (cnt >= 2).sum(1)
            totp[c0:c0 + c] += (cnt * (cnt - 1) // 2).sum(1)
    return tot2, totp


def pvals(obs, null):
    up = (1 + int((null >= obs).sum())) / (B + 1)
    lo = (1 + int((null <= obs).sum())) / (B + 1)
    return up, min(1.0, 2 * min(up, lo))


def analyse(coll, recs, reqs_by_review, level, jev_p):
    S = review_structs(recs, reqs_by_review, level)
    per = {}
    O2 = Op = 0; E2 = Ep = 0.0
    for k, s in S.items():
        o2, op = stats_for(s["y"], s["gid"], len(s["sizes"]))
        e2, ep = expected(s["N"], s["K"], s["sizes"])
        per[k] = {"N": s["N"], "K": s["K"], "n_requests": int((s["sizes"] > 0).sum()), "obs_req_ge2": o2, "exp_req_ge2": e2,
                  "obs_pairs": op, "exp_pairs": ep, "oe_req_ge2": (o2 / e2) if e2 > 0 else None, "oe_pairs": (op / ep) if ep > 0 else None}
        O2 += o2; Op += op; E2 += e2; Ep += ep
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    n2, npairs = permute(S, rng)
    p2 = pvals(O2, n2); pp = pvals(Op, npairs)
    oe = [v["oe_req_ge2"] for v in per.values() if v["oe_req_ge2"] is not None]
    oep = [v["oe_pairs"] for v in per.values() if v["oe_pairs"] is not None]
    # (C) score of included records by co-batching (descriptive, confounded)
    diffs = []
    for k, s in S.items():
        if s["K"] < 2:
            continue
        cnt = np.bincount(s["gid"], weights=s["y"], minlength=len(s["sizes"]))
        pos = np.where(s["y"] == 1)[0]
        co = [jev_p[s["ids"][i]] for i in pos if cnt[s["gid"][i]] >= 2]
        so = [jev_p[s["ids"][i]] for i in pos if cnt[s["gid"][i]] < 2]
        if co and so:
            diffs.append({"review": k, "n_cobatched": len(co), "n_solo": len(so), "mean_cobatched": float(np.mean(co)),
                          "mean_solo": float(np.mean(so)), "diff": float(np.mean(co) - np.mean(so))})
    return {
        "level": level, "n_reviews": len(S), "n_reviews_K_ge_2": sum(1 for s in S.values() if s["K"] >= 2),
        "records": sum(s["N"] for s in S.values()), "included": sum(s["K"] for s in S.values()),
        "requests": sum(int((s["sizes"] > 0).sum()) for s in S.values()),
        "observed_requests_ge2": O2, "expected_requests_ge2": E2, "oe_requests_ge2": O2 / E2 if E2 else None,
        "perm_mean_requests_ge2": float(n2.mean()), "perm_sd_requests_ge2": float(n2.std(ddof=1)),
        "perm_q025_q975_requests_ge2": [float(np.percentile(n2, 2.5)), float(np.percentile(n2, 97.5))],
        "p_one_sided_excess_requests_ge2": p2[0], "p_two_sided_requests_ge2": p2[1],
        "observed_pairs": Op, "expected_pairs": Ep, "oe_pairs": Op / Ep if Ep else None,
        "perm_mean_pairs": float(npairs.mean()), "p_one_sided_excess_pairs": pp[0], "p_two_sided_pairs": pp[1],
        "per_review_oe_requests_ge2": {"n": len(oe), "median": float(np.median(oe)) if oe else None,
                                       "min": float(min(oe)) if oe else None, "max": float(max(oe)) if oe else None,
                                       "n_above_1": sum(1 for x in oe if x > 1), "n_below_1": sum(1 for x in oe if x < 1)},
        "per_review_oe_pairs": {"n": len(oep), "median": float(np.median(oep)) if oep else None,
                                "min": float(min(oep)) if oep else None, "max": float(max(oep)) if oep else None},
        "score_cobatched_minus_solo": {"n_reviews": len(diffs), "mean_of_within_review_diff": float(np.mean([d["diff"] for d in diffs])) if diffs else None,
                                       "median": float(np.median([d["diff"] for d in diffs])) if diffs else None,
                                       "n_positive": sum(1 for d in diffs if d["diff"] > 0), "n_negative": sum(1 for d in diffs if d["diff"] < 0),
                                       "per_review": diffs},
        "permutation_seconds": time.time() - t0,
        "per_review": per,
    }


def main():
    res = {"analysis_id": "AN-0002-01", "post_hoc": True, "seed": SEED, "permutations": B,
           "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "collections": {}}
    for coll, (rf, jf) in COLL.items():
        log("collection", coll)
        recs = json.load(open(rf)); jev = json.load(open(jf))
        jev_p = {e["id"]: float(e["p"]) for e in jev}
        reqs = reconstruct_requests(jev)
        # diagnostics of the reconstruction
        size_mismatch = sum(1 for s in reqs if len(s) != int(s[0].get("batch") or 1))
        pos_in_file = {}
        byrev_ids = defaultdict(list)
        for r in recs:
            byrev_ids[r["review"]].append(r["id"])
        for k, ids in byrev_ids.items():
            for i, x in enumerate(ids):
                pos_in_file[x] = i
        noncontig = sum(1 for s in reqs if sorted(pos_in_file[e["id"]] for e in s) != list(range(min(pos_in_file[e["id"]] for e in s), min(pos_in_file[e["id"]] for e in s) + len(s))))
        # deterministic slices of ten in file order
        det = {}
        for k, ids in byrev_ids.items():
            for i in range(0, len(ids), 10):
                key = frozenset(ids[i:i + 10])
                det[key] = k
        actual_sets = [frozenset(e["id"] for e in s) for s in reqs]
        n_equal_det = sum(1 for a in actual_sets if a in det)
        rec_in_equal = sum(len(a) for a in actual_sets if a in det)
        reqs_by_review = defaultdict(list)
        for s in reqs:
            reqs_by_review[s[0]["review"]].append(s)
        # deterministic grouping as a secondary structure
        det_by_review = {k: [[{"id": x} for x in ids[i:i + 10]] for i in range(0, len(ids), 10)] for k, ids in byrev_ids.items()}
        # keep review order of the data file
        reqs_by_review = {k: reqs_by_review[k] for k in byrev_ids}
        C = {"records_file": rf, "jev_file": jf, "n_records": len(recs), "n_requests_reconstructed": len(reqs),
             "request_size_distribution": dict(Counter(len(s) for s in reqs)),
             "requests_size_not_equal_batch_field": size_mismatch, "requests_not_contiguous_in_file_order": noncontig,
             "requests_equal_to_deterministic_slice": n_equal_det, "records_in_requests_equal_to_deterministic_slice": rec_in_equal,
             "n_deterministic_slices": len(det)}
        log("  requests", len(reqs), "size mismatch", size_mismatch, "noncontig", noncontig, "equal det", n_equal_det)
        C["order"] = order_checks(coll, recs)
        C["position_auc"] = {}
        for level in ["label", "label_ta"]:
            pa = position_auc(recs, level)
            if not pa:
                continue
            aucs = [v["auc_position"] for v in pa.values()]
            C["position_auc"][level] = {"n_reviews": len(pa), "median": float(np.median(aucs)), "min": float(min(aucs)), "max": float(max(aucs)),
                                        "n_p_lt_0.05": sum(1 for v in pa.values() if v["p"] < 0.05),
                                        "n_p_lt_0.05_auc_gt_0.5": sum(1 for v in pa.values() if v["p"] < 0.05 and v["auc_position"] > 0.5),
                                        "n_p_lt_0.05_auc_lt_0.5": sum(1 for v in pa.values() if v["p"] < 0.05 and v["auc_position"] < 0.5),
                                        "per_review": pa}
        C["batch"] = {}
        for level in ["label", "label_ta"]:
            if not any(lab(r[level]) is not None for r in recs):
                continue
            log("  batch composition", level)
            C["batch"][level] = analyse(coll, recs, reqs_by_review, level, jev_p)
            log("   obs", C["batch"][level]["observed_requests_ge2"], "exp", round(C["batch"][level]["expected_requests_ge2"], 2),
                "p1", C["batch"][level]["p_one_sided_excess_requests_ge2"])
            log("  deterministic slices", level)
            C["batch"][level + "_deterministic_slices"] = analyse(coll, recs, det_by_review, level, jev_p)
        res["collections"][coll] = C
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
    # compact summary table
    rows = []
    for coll, C in res["collections"].items():
        for lv, b in C["batch"].items():
            rows.append({"collection": coll, "level": lv, "reviews": b["n_reviews"], "reviews_K_ge_2": b["n_reviews_K_ge_2"],
                         "records": b["records"], "included": b["included"], "requests": b["requests"],
                         "obs_req_ge2": b["observed_requests_ge2"], "exp_req_ge2": round(b["expected_requests_ge2"], 3),
                         "oe_req_ge2": round(b["oe_requests_ge2"], 4) if b["oe_requests_ge2"] else None,
                         "p1_req": b["p_one_sided_excess_requests_ge2"], "p2_req": b["p_two_sided_requests_ge2"],
                         "obs_pairs": b["observed_pairs"], "exp_pairs": round(b["expected_pairs"], 3),
                         "oe_pairs": round(b["oe_pairs"], 4) if b["oe_pairs"] else None,
                         "p1_pairs": b["p_one_sided_excess_pairs"], "p2_pairs": b["p_two_sided_pairs"],
                         "score_diff_mean": b["score_cobatched_minus_solo"]["mean_of_within_review_diff"],
                         "score_diff_reviews": b["score_cobatched_minus_solo"]["n_reviews"]})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "batch_composition_summary.csv"), index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
