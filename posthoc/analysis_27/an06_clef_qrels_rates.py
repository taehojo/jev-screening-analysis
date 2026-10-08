"""AN-0002-06 (round 2, post hoc, descriptive): per review type inclusion rates of the abstract-level and
content-level relevance files (qrels) of CLEF 2019 TAR Task 2 (Testing), with file names, recomputed from
the public files in clef/tar (git clone of github.com/CLEF-TAR/tar), compared with the stored values of
AN-0001-19/lgar_label_level_check.json and with the rates printed in LGAR Table 1.

Counting rule of AN-0001-19 (kept as the primary count so that the comparison is like for like): every line
with at least four fields is one record of its topic; the fourth field is the relevance (0 or 1).
Additional checks: distinct relevance values, duplicate (topic, document) lines, and the rates with duplicates
counted once. Public data only; no model call.
"""
import collections
import hashlib
import json
import os
import subprocess

ROOT = "/N/project/AiLab/jev"
BASE = f"{ROOT}/clef/tar/2019-TAR/Task2/Testing"
OUT = os.path.dirname(os.path.abspath(__file__))
STORED = f"{ROOT}/review_pipeline/rounds/round_0001/revision/analysis/AN-0001-19/lgar_label_level_check.json"
LGAR = {"DTA": 7.05, "Intervention": 5.49, "Qualitative": 1.32, "Prognosis": 5.70}
LGAR_N = {"DTA": (8, 30521), "Intervention": (20, 41996), "Qualitative": (2, 6536), "Prognosis": (1, 3367)}
TYPES = ["DTA", "Intervention", "Prognosis", "Qualitative"]


def git(*a):
    return subprocess.run(["git", "-C", f"{ROOT}/clef/tar", *a], capture_output=True, text=True).stdout.strip()


def main():
    stored = json.load(open(STORED))["qrels"]
    res = {"analysis_id": "AN-0002-06", "post_hoc": True,
           "repository": git("remote", "get-url", "origin"), "commit": git("log", "-1", "--format=%H %cd"),
           "files": {}, "by_type": {}, "comparison": {}}
    for kind in ["abs", "content"]:
        res["by_type"][kind] = {}
        for typ in TYPES:
            fn = f"{BASE}/{typ}/qrels/full.test.{typ.lower()}.{kind}.2019.qrels"
            raw = open(fn, "rb").read()
            rel = os.path.relpath(fn, f"{ROOT}/clef/tar")
            tot, pos = collections.Counter(), collections.Counter()
            seen = collections.Counter()
            vals = collections.Counter()
            udoc = collections.defaultdict(dict)
            nlines = 0
            for line in raw.decode().splitlines():
                p = line.split()
                if len(p) < 4:
                    continue
                nlines += 1
                t, d, r = p[0], p[2], int(p[3])
                vals[r] += 1
                tot[t] += 1
                pos[t] += r
                seen[(t, d)] += 1
                udoc[t][d] = max(udoc[t].get(d, 0), r)
            rates = [pos[t] / tot[t] for t in tot]
            urates = [sum(udoc[t].values()) / len(udoc[t]) for t in udoc]
            dups = {f"{k[0]}": v for k, v in seen.items() if v > 1}
            rec = {"file": rel, "sha256": hashlib.sha256(raw).hexdigest(), "n_topics": len(tot),
                   "n_records": sum(tot.values()), "n_relevant": sum(pos.values()),
                   "macro_rate_pct": 100 * sum(rates) / len(rates),
                   "pooled_rate_pct": 100 * sum(pos.values()) / sum(tot.values()),
                   "relevance_values": dict(vals),
                   "n_duplicate_topic_document_pairs": sum(1 for v in seen.values() if v > 1),
                   "duplicate_lines_by_topic": dict(collections.Counter(k[0] for k, v in seen.items() if v > 1)),
                   "n_records_unique": sum(len(v) for v in udoc.values()),
                   "n_relevant_unique": sum(sum(v.values()) for v in udoc.values()),
                   "macro_rate_unique_pct": 100 * sum(urates) / len(urates),
                   "per_topic": {t: {"records": tot[t], "relevant": pos[t]} for t in sorted(tot)}}
            res["by_type"][kind][typ] = rec
            s = stored[kind][typ]
            res["comparison"].setdefault(kind, {})[typ] = {
                "stored_macro_pct": s["macro_incl_rate_pct"], "recomputed_macro_pct_2dp": round(rec["macro_rate_pct"], 2),
                "stored_matches": round(rec["macro_rate_pct"], 2) == s["macro_incl_rate_pct"]
                and s["papers"] == rec["n_records"] and s["positives"] == rec["n_relevant"] and s["n_slr"] == rec["n_topics"],
                "lgar_table1_pct": LGAR[typ], "lgar_topics_records": LGAR_N[typ],
                "equals_lgar_2dp": round(rec["macro_rate_pct"], 2) == LGAR[typ],
                "topics_records_equal_lgar": (rec["n_topics"], rec["n_records"]) == LGAR_N[typ],
                "unique_records_equal_lgar": rec["n_records_unique"] == LGAR_N[typ][1]}
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    # small table for the appendix
    lines = ["| Review type | Relevance file | Topics | Records | Relevant | Macro-averaged rate (%) | Pooled rate (%) | LGAR table 1 (%) |",
             "|---|---|---|---|---|---|---|---|"]
    for typ in TYPES:
        for kind in ["abs", "content"]:
            r = res["by_type"][kind][typ]
            lines.append(f"| {typ} | {os.path.basename(r['file'])} | {r['n_topics']} | {r['n_records']} | {r['n_relevant']} | "
                         f"{r['macro_rate_pct']:.2f} | {r['pooled_rate_pct']:.2f} | {LGAR[typ]:.2f} |")
    open(os.path.join(OUT, "an06_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(res["comparison"], indent=1))


if __name__ == "__main__":
    main()
