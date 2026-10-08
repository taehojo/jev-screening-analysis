"""AN-0002-02 (round 2, post hoc, descriptive).

Part A. Every comparison of table S17 (appendix v1) that concerns reliability (indicator of recall >= 0.95 per
review) is recomputed from the per-review files of round 1, as a paired binary comparison:
  discordant counts b (only the first rule reliable) and c (only the second rule reliable);
  McNemar asymptotic test without continuity correction, z = (b - c) / sqrt(b + c), two-sided p;
  McNemar exact conditional test, p = min(1, 2 * P(X <= min(b, c))), X ~ Binomial(b + c, 1/2);
  McNemar mid-p, p_exact - P(X = b) (or 1 - 0.5 * P(X = b) when b = c);
  95% CI of the difference in proportions (first minus second) by the Tango asymptotic score method and by the
  Newcombe square-and-add (hybrid score, method 10) method.
The formulas follow the R source of the CRAN package contingencytables 3.1.0 (Fagerland, Lydersen and Laake),
saved in ../AN-0002-04/sources/ (the R package was not installed; there is no R on this server). The Python
implementation is checked against the expected values in that package's tests (tests/testthat/test-ch8.R).
Tests are reported only when b + c > 0. The stored percentile bootstrap intervals and stored Wilcoxon p values
of round 1 are shown next to the new values; the recomputed asymptotic McNemar p is compared with the stored
Wilcoxon p.

Part B. Knee method on the hybrid ranking in reviews with fewer than 150 and at least 150 records, and fewer than
500 and at least 500 records, for held-out, CLEF and development reviews, with the definitions of table S18 and
of AN-0001-04 size_restriction(): reliability judged on the seed-mean recall (column *_knee_reliable), Wilson 95%
CI (z = 1.959963984540054), macro proportion read (mean of *_knee_work), pooled proportion read
(sum(work * N) / sum(N)), lowest recall (min of *_knee_rec). The threshold rule is given for the same strata.
Stored values in AN-0001-04/results.json (size_restriction_150/500) are compared where they exist.

Inputs are read only. No random numbers, no model call.
"""
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import binom, norm

OUT = os.path.dirname(os.path.abspath(__file__))
R1 = "/N/project/AiLab/jev/review_pipeline/rounds/round_0001/revision/analysis"
Z = 1.959963984540054


# ---------------------------------------------------------------- methods (from contingencytables R source)
def mcnemar_asymptotic(b, c):
    nd = b + c
    if nd == 0:
        return None, 1.0
    z = (b - c) / math.sqrt(nd)
    return z, 2 * (1 - norm.cdf(abs(z)))


def mcnemar_exact_cond(b, c):
    return min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5))


def mcnemar_midp(b, c):
    if b == c:
        return 1 - 0.5 * binom.pmf(b, b + c, 0.5)
    return min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5)) - binom.pmf(b, b + c, 0.5)


def wilson(x, n, z=Z):
    est = x / n
    A = (2 * n * est + z ** 2) / (2 * n + 2 * z ** 2)
    B = (z * math.sqrt(z ** 2 + 4 * n * est * (1 - est))) / (2 * n + 2 * z ** 2)
    return A - B, A + B


def tango_ci(n11, n12, n21, n22, z=Z):
    N = n11 + n12 + n21 + n22
    est = (n12 - n21) / N
    tol = 1e-7

    def p21t(d0):
        A = 2 * N
        B = -n12 - n21 + (2 * N - n12 + n21) * d0
        C = -n21 * d0 * (1 - d0)
        return (math.sqrt(B * B - 4 * A * C) - B) / (2 * A)

    def T(d0):
        return (n12 - n21 - N * d0) / math.sqrt(N * (2 * p21t(d0) + d0 * (1 - d0)))

    L = -1.0 if est == -1 else brentq(lambda d: T(d) - z, -1 + tol, 1 - tol, xtol=tol)
    U = 1.0 if est == 1 else brentq(lambda d: T(d) + z, -1 + tol, 1 - tol, xtol=tol)
    return est, L, U


def newcombe_ci(n11, n12, n21, n22, z=Z):
    N = n11 + n12 + n21 + n22
    r1, r2 = n11 + n12, n21 + n22   # row totals: first rule reliable / not
    c1, c2 = n11 + n21, n12 + n22   # column totals: second rule reliable / not
    p1, p2 = r1 / N, c1 / N
    est = p1 - p2
    l1, u1 = wilson(r1, N, z)
    l2, u2 = wilson(c1, N, z)
    if r1 == 0 or r2 == 0 or c1 == 0 or c2 == 0:
        psi = 0.0
    else:
        nprod = r1 * r2 * c1 * c2
        A = n11 * n22 - n12 * n21
        if A > N / 2:
            psi = (A - N / 2) / math.sqrt(nprod)
        elif 0 <= A <= N / 2:
            psi = 0.0
        else:
            psi = A / math.sqrt(nprod)
    L = est - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * psi * (p1 - l1) * (u2 - p2))
    U = est + math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2 - 2 * psi * (p2 - l2) * (u1 - p1))
    return est, L, U, psi


def validate():
    """Expected values from tests/testthat/test-ch8.R of contingencytables 3.1.0 (printed to 4 or 6 decimals)."""
    cavo = (59, 6, 16, 80)   # cavo_2012 <- rbind(c(59, 6), c(16, 80)) in R/datasets.R
    out = []

    def chk(name, got, exp, dp):
        ok = round(got, dp) == round(exp, dp) or abs(got - exp) < 0.5 * 10 ** (-dp) + 1e-12
        out.append({"check": name, "got": got, "expected": exp, "decimals": dp, "ok": bool(ok)})

    zz, p = mcnemar_asymptotic(cavo[1], cavo[2])
    chk("asymptotic P cavo", p, 0.033006, 6)
    chk("asymptotic Z cavo", zz, -2.132, 3)
    chk("exact conditional P cavo", mcnemar_exact_cond(cavo[1], cavo[2]), 0.052479, 6)
    chk("mid-P cavo", mcnemar_midp(cavo[1], cavo[2]), 0.034690, 6)
    e, L, U = tango_ci(*cavo)
    chk("Tango estimate cavo", e, -0.0621, 4); chk("Tango lower cavo", L, -0.1240, 4); chk("Tango upper cavo", U, -0.0054, 4)
    e, L, U = tango_ci(0, 3, 0, 0)     # matrix(c(0, 0, 3, 0), 2) is column-major: n[1,2] = 3
    chk("Tango estimate (0,3,0,0)", e, 1.0, 4); chk("Tango lower (0,3,0,0)", L, -0.1230, 4); chk("Tango upper (0,3,0,0)", U, 1.0, 4)
    e, L, U = tango_ci(0, 0, 3, 0)     # matrix(c(0, 3, 0, 0), 2): n[2,1] = 3
    chk("Tango estimate (0,0,3,0)", e, -1.0, 4); chk("Tango lower (0,0,3,0)", L, -1.0, 4); chk("Tango upper (0,0,3,0)", U, 0.1230, 4)
    e, L, U, _ = newcombe_ci(*cavo)
    chk("Newcombe estimate cavo", e, -0.0621, 4); chk("Newcombe lower cavo", L, -0.1186, 4); chk("Newcombe upper cavo", U, -0.0046, 4)
    e, L, U, _ = newcombe_ci(1, 0, 0, 1)
    chk("Newcombe lower diag(1,1)", L, -0.5734, 4); chk("Newcombe upper diag(1,1)", U, 0.5734, 4)
    return out


# ---------------------------------------------------------------- data
def load():
    P8 = pd.read_csv(f"{R1}/AN-0001-08/per_review_stopping.csv")
    P4 = {c: pd.read_csv(f"{R1}/AN-0001-04/per_review_{f}.csv") for c, f in
          [("heldout", "heldout"), ("clef", "clef"), ("development", "development")]}
    for c in P4:
        P4[c] = P4[c][P4[c].thr_work.notna()].set_index("review")
    return P8, P4


ROWS8 = [  # (S17 label, first column, second column, stored key in AN-0001-08 reliability_differences or None)
    ("Statistical criterion on the Jev ranking minus threshold", "jev_stat_rec", "thr_rec", "jev_stat_minus_thr"),
    ("Statistical criterion minus knee, ASReview ranking", "asr_stat_rec", "asr_knee_rec", "asr_stat_minus_asr_knee"),
    ("Statistical criterion, Jev ranking minus ASReview ranking", "jev_stat_rec", "asr_stat_rec", "jev_stat_minus_asr_stat"),
    ("Threshold then statistical criterion minus statistical criterion alone, Jev ranking", "jev_comb_rec", "jev_stat_rec", None),
    ("Statistical criterion, hybrid ranking minus ASReview ranking", "hybrid_stat_rec", "asr_stat_rec", None),
    ("Statistical criterion minus knee, hybrid ranking", "hybrid_stat_rec", "hybrid_knee_rec", None),
    ("Threshold minus statistical criterion on the ASReview ranking", "thr_rec", "asr_stat_rec", "thr_minus_asr_stat"),
    ("Threshold then statistical criterion on the Jev ranking minus threshold", "jev_comb_rec", "thr_rec", "jev_comb_minus_thr"),
]
ROWS4 = [("Threshold minus knee on the ASReview ranking", "thr_reliable", "asr_knee_reliable", "paired_threshold_vs_knee_asr"),
         ("Threshold minus knee on the hybrid ranking", "thr_reliable", "hybrid_knee_reliable", "paired_threshold_vs_knee_hybrid")]


def compare(label, coll, a, b, source, stored_ci=None, stored_p=None, stored_counts=None):
    a = np.asarray(a, int); b_ = np.asarray(b, int)
    n11 = int(((a == 1) & (b_ == 1)).sum()); n12 = int(((a == 1) & (b_ == 0)).sum())
    n21 = int(((a == 0) & (b_ == 1)).sum()); n22 = int(((a == 0) & (b_ == 0)).sum())
    N = n11 + n12 + n21 + n22
    r = {"collection": coll, "comparison": label, "source": source, "n_reviews": N,
         "first_reliable": n11 + n12, "second_reliable": n11 + n21,
         "n11_both": n11, "b_only_first": n12, "c_only_second": n21, "n22_neither": n22,
         "difference": (n12 - n21) / N}
    if n12 + n21 > 0:
        z, p = mcnemar_asymptotic(n12, n21)
        r.update({"mcnemar_z": z, "p_asymptotic": p, "p_exact_conditional": mcnemar_exact_cond(n12, n21),
                  "p_mid": mcnemar_midp(n12, n21)})
    else:
        r.update({"mcnemar_z": None, "p_asymptotic": None, "p_exact_conditional": None, "p_mid": None,
                  "test_note": "no discordant reviews; no test"})
    e, L, U = tango_ci(n11, n12, n21, n22)
    r.update({"tango_lower": L, "tango_upper": U})
    e2, L2, U2, psi = newcombe_ci(n11, n12, n21, n22)
    r.update({"newcombe_lower": L2, "newcombe_upper": U2, "newcombe_psi": psi})
    r["stored_bootstrap_ci"] = stored_ci
    r["stored_wilcoxon_p"] = stored_p
    if stored_p is not None and r["p_asymptotic"] is not None:
        r["asymptotic_minus_stored_wilcoxon_p"] = r["p_asymptotic"] - stored_p
    if stored_counts is not None:
        r["stored_counts_b_c"] = stored_counts
        r["counts_match_stored"] = tuple(stored_counts) == (n12, n21)
    return r


def part_a(P8, P4):
    R8 = json.load(open(f"{R1}/AN-0001-08/results.json"))["collections"]
    R4 = json.load(open(f"{R1}/AN-0001-04/results.json"))["collections"]
    rows = []
    for coll in ["heldout", "clef"]:
        T = P8[P8.collection == coll].set_index("review")
        near = {c: [k for k in T.index if abs(T.loc[k, c] - 0.95) < 1e-9] for c in T.columns if c.endswith("_rec")}
        assert not any(near.values()), near
        T4 = P4[coll]
        for label, ca, cb, key in ROWS4:
            revs = list(T4.index)
            st = R4[coll].get(key)
            rows.append(compare(label, coll, T4.loc[revs, ca], T4.loc[revs, cb], f"AN-0001-04/per_review_{coll}.csv",
                                stored_ci=st["reliability_diff_boot95"] if st else None, stored_p=None,
                                stored_counts=[st["reviews_threshold_reliable_knee_not"], st["reviews_knee_reliable_threshold_not"]] if st else None))
        for label, ca, cb, key in ROWS8:
            revs = list(T.index)
            st = R8[coll]["reliability_differences"].get(key) if key else None
            rows.append(compare(label, coll, (T.loc[revs, ca] >= 0.95).astype(int), (T.loc[revs, cb] >= 0.95).astype(int),
                                "AN-0001-08/per_review_stopping.csv",
                                stored_ci=st["ci95"] if st else None, stored_p=st.get("wilcoxon_p") if st else None,
                                stored_counts=[st["wins"], st["losses"]] if st else None))
    T4 = P4["development"]
    st = R4["development"]["paired_threshold_vs_knee_asr"]
    rows.append(compare("Threshold minus knee on the ASReview ranking", "development", T4.thr_reliable, T4.asr_knee_reliable,
                        "AN-0001-04/per_review_development.csv", stored_ci=st["reliability_diff_boot95"], stored_p=None,
                        stored_counts=[st["reviews_threshold_reliable_knee_not"], st["reviews_knee_reliable_threshold_not"]]))
    # cross-check with the exact-arithmetic recount of round 1 (fx01_exact_paired.csv), where it exists
    fx = pd.read_csv(f"{R1}/audit_fix_iter2/fx01_exact_paired.csv")
    fxr = fx[fx.key.str.contains("reliability")]
    return rows, fxr


def part_b(P4):
    R4 = json.load(open(f"{R1}/AN-0001-04/results.json"))["collections"]
    out = []
    for coll in ["heldout", "clef", "development"]:
        T = P4[coll]
        for cut in [150, 500]:
            for lab, S in [(f"N<{cut}", T[T.N < cut]), (f"N>={cut}", T[T.N >= cut])]:
                r = {"collection": coll, "stratum": lab, "n_reviews": int(len(S))}
                if len(S):
                    for tag, rel, work, rec in [("knee_hybrid", "hybrid_knee_reliable", "hybrid_knee_work", "hybrid_knee_rec"),
                                                ("threshold", "thr_reliable", "thr_work", "thr_rec"),
                                                ("knee_asr", "asr_knee_reliable", "asr_knee_work", "asr_knee_rec")]:
                        k = int(S[rel].sum()); lo, hi = wilson(k, len(S)) if True else (None, None)
                        r[tag] = {"reliable_k": k, "n": int(len(S)), "wilson95": [max(0.0, lo), min(1.0, hi)],
                                  "read_macro": float(S[work].mean()),
                                  "read_pooled": float((S[work] * S.N).sum() / S.N.sum()),
                                  "lowest_recall": float(S[rec].min()),
                                  "records": int(S.N.sum())}
                    st = R4[coll].get(f"size_restriction_{cut}", {}).get(lab)
                    if st and "hybrid_knee_reliable" in st:
                        r["stored_knee_hybrid"] = {"k": st["hybrid_knee_reliable"]["k"], "macro": st["hybrid_knee_work_macro"],
                                                   "pooled": st["hybrid_knee_work_pooled"]}
                        r["stored_matches"] = (st["hybrid_knee_reliable"]["k"] == r["knee_hybrid"]["reliable_k"]
                                               and abs(st["hybrid_knee_work_macro"] - r["knee_hybrid"]["read_macro"]) < 1e-12
                                               and abs(st["hybrid_knee_work_pooled"] - r["knee_hybrid"]["read_pooled"]) < 1e-12)
                    else:
                        r["stored_knee_hybrid"] = None
                out.append(r)
    return out


def main():
    val = validate()
    assert all(v["ok"] for v in val), [v for v in val if not v["ok"]]
    P8, P4 = load()
    rows, fxr = part_a(P8, P4)
    strata = part_b(P4)
    res = {"analysis_id": "AN-0002-02", "post_hoc": True, "z": Z,
           "method_source": "R source of CRAN contingencytables 3.1.0 (saved in ../AN-0002-04/sources/cran_contingencytables_*.R)",
           "validation_against_package_tests": val,
           "part_a_reliability_comparisons": rows,
           "part_a_round1_exact_recount_rows": fxr[["key", "exact_higher", "exact_lower", "exact_tied", "exact_p"]].to_dict(orient="records"),
           "part_b_knee_hybrid_by_size": strata}
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
    pd.DataFrame(rows).drop(columns=["stored_bootstrap_ci", "stored_counts_b_c"], errors="ignore").to_csv(os.path.join(OUT, "table_s17_mcnemar.csv"), index=False)
    flat = []
    for s in strata:
        base = {"collection": s["collection"], "stratum": s["stratum"], "n_reviews": s["n_reviews"]}
        for tag in ["knee_hybrid", "threshold", "knee_asr"]:
            if tag in s:
                d = s[tag]
                base.update({f"{tag}_reliable": f"{d['reliable_k']} of {d['n']}", f"{tag}_wilson_lo": d["wilson95"][0], f"{tag}_wilson_hi": d["wilson95"][1],
                             f"{tag}_read_macro": d["read_macro"], f"{tag}_read_pooled": d["read_pooled"], f"{tag}_lowest_recall": d["lowest_recall"]})
        base["stored_matches"] = s.get("stored_matches")
        flat.append(base)
    pd.DataFrame(flat).to_csv(os.path.join(OUT, "table_s18_knee_hybrid_by_size.csv"), index=False)
    # print
    for r in rows:
        print(f"{r['collection']:11s} {r['comparison'][:70]:70s} b={r['b_only_first']} c={r['c_only_second']} d={r['difference']:+.3f} "
              f"z={r['mcnemar_z'] if r['mcnemar_z'] is None else round(r['mcnemar_z'],3)} pA={r['p_asymptotic']} pE={r['p_exact_conditional']} pM={r['p_mid']} "
              f"Tango=({r['tango_lower']:.3f},{r['tango_upper']:.3f}) Newc=({r['newcombe_lower']:.3f},{r['newcombe_upper']:.3f}) "
              f"stored_p={r['stored_wilcoxon_p']} counts_match={r.get('counts_match_stored')}")
    print(fxr[["key", "exact_higher", "exact_lower", "exact_p"]].to_string())
    print(pd.DataFrame(flat).to_string())


if __name__ == "__main__":
    main()
