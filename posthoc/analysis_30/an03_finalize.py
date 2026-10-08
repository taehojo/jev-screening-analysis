"""AN-0003-03 finalize: exact line of the saved copies that contains the sentence, with file hashes and times.
No network. Reads the saved copies and results_step1.json; writes results.json."""
import datetime, hashlib, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
s1 = json.load(open(os.path.join(HERE, "results_step1.json")))
out = {"analysis_id": "AN-0003-03", "post_hoc": True, "network_used": False,
       "step2_cdx_needed": False, "step2_reason": "the sentence is present in the capture of Sept 26, 2026 (20260926091045); the plan queries later captures only if it is absent",
       "copies": {}}
for key, c in s1["copies"].items():
    lines = open(c["text"], encoding="utf-8").read().splitlines()
    hit = [(i + 1, l.strip()) for i, l in enumerate(lines) if "subsidi" in l.lower()]
    out["copies"][key] = {"description": c["description"], "text_file": c["text"], "html_file": c["html"],
                          "sha256_html": hashlib.sha256(open(c["html"], "rb").read()).hexdigest(),
                          "sha256_text": hashlib.sha256(open(c["text"], "rb").read()).hexdigest(),
                          "mtime_html": datetime.datetime.fromtimestamp(os.path.getmtime(c["html"])).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                          "lines_with_subsidi": [{"line": n, "text": t} for n, t in hit],
                          "counts_step1": {src: c["matches"][src]["_counts"] for src in c["matches"]}}
texts = {k: [h["text"] for h in v["lines_with_subsidi"]] for k, v in out["copies"].items()}
out["same_line_in_all_copies"] = len({tuple(v) for v in texts.values()}) == 1
out["quotation"] = texts["wayback_20260926091045"][0] if texts["wayback_20260926091045"] else None
out["quotation_source"] = "https://web.archive.org/web/20260926091045/https://typesafe.ai/blog/introducing-system-one-models-and-jev (captured 2026-09-26 09:10:45 UTC; saved in rounds/round_0002/revision/analysis/AN-0002-03/snapshots/blog_20260926091045_id_decoded.txt, line %d)" % out["copies"]["wayback_20260926091045"]["lines_with_subsidi"][0]["line"]
out["note_typography"] = "apostrophes in the source are U+2019; the source spelling is 'subsidized'"
out["finished"] = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: out[k] for k in ("same_line_in_all_copies", "quotation", "quotation_source")}, ensure_ascii=False, indent=1))
for k, v in out["copies"].items(): print(k, [h["line"] for h in v["lines_with_subsidi"]], v["sha256_html"][:16])
