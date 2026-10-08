"""AN-0002-03 step 2: decode and compare the archived copies of reference 16.

The id_ snapshots fetched by an03_archive_ref16.py are the archived HTTP bodies with their original gzip content
encoding (first two bytes 1f 8b), so the text checks in results_fetch.json (made on the compressed bytes) are void.
This step gunzips them, extracts the text of the live copy (AN-0001-15, fetched 2026-09-26 19:00:40 EDT) and of
both snapshots with the same method (UTF-8 decoding, scripts and styles removed, one line per block element),
checks the reference-model sentence and the date, compares the files byte for byte and the texts line by line,
and adds the model periods of the texts sent to the evaluated models (from results_fetch.json, which read
audit_fix_iter4/fx41_text_provenance.json). Writes results.json. No network access.
"""
import difflib
import gzip
import hashlib
import json
import os
import re

import lxml.html

OUT = os.path.dirname(os.path.abspath(__file__))
F = json.load(open(os.path.join(OUT, "results_fetch.json")))
LIVE = F["live_copy"]["html"]
SENT = "We use the average of GPT-6 Astra and Fable 5.1 as the reference answer"
DATE = "Sep 15, 2026"
BLOCK = {"p", "h1", "h2", "h3", "h4", "h5", "li", "td", "th", "div", "section", "article", "header", "footer", "title", "span", "a", "button"}


def text_lines(raw):
    html = raw.decode("utf-8")
    doc = lxml.html.document_fromstring(html)
    for bad in doc.xpath("//script|//style|//noscript|//svg"):
        bad.getparent().remove(bad)
    for el in doc.iter():
        if isinstance(el.tag, str) and el.tag in {"p", "h1", "h2", "h3", "h4", "h5", "li", "td", "th", "title", "br", "div"}:
            el.tail = ("\n" + el.tail) if el.tail else "\n"
            if el.text:
                el.text = "\n" + el.text
    txt = doc.text_content()
    return [l for l in (" ".join(x.split()) for x in txt.splitlines()) if l]


def main():
    live_raw = open(LIVE, "rb").read()
    live_lines = text_lines(live_raw)
    res = {"analysis_id": "AN-0002-03", "reference": "16", "live_url": F["page"],
           "live_copy": {"file": LIVE, "fetched_et": "2026-09-26 19:00:40 EDT (AN-0001-15/sources/fetch_record.txt)",
                         "bytes": len(live_raw), "sha256": hashlib.sha256(live_raw).hexdigest(),
                         "sentence_present": SENT in " ".join(live_lines), "date_present": DATE in " ".join(live_lines),
                         "sentence_line": [l for l in live_lines if "GPT-6 Astra" in l]},
           "cdx": {k: F[k] for k in ["cdx_source", "cdx_n_rows", "cdx_n_status200", "cdx_first", "cdx_last", "cdx_n_distinct_digests_status200"]},
           "cdx_requery_attempts": F["cdx_query"]["attempts"], "snapshots": {}}
    for label, s in F["snapshots"].items():
        raw = open(s["saved_as"], "rb").read()
        dec = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
        fn = s["saved_as"].replace("_id.html", "_id_decoded.html")
        open(fn, "wb").write(dec)
        lines = text_lines(dec)
        open(fn.replace(".html", ".txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
        diff = list(difflib.unified_diff(live_lines, lines, fromfile="live_copy_2026-09-26_190040_EDT",
                                         tofile=f"snapshot_{s['cdx_timestamp']}", lineterm="", n=0))
        dfn = os.path.join(OUT, f"diff_text_live_vs_{s['cdx_timestamp']}.txt")
        open(dfn, "w", encoding="utf-8").write("\n".join(diff) + "\n")
        joined = " ".join(lines)
        m = re.search(r"(Sep \d{1,2}, 2026)", joined)
        res["snapshots"][label] = {
            "cdx_timestamp_utc": s["cdx_timestamp"], "snapshot_url": s["snapshot_url"], "fetched_url": s["fetched_url"],
            "http_status": s["http_status"], "fetch_time": s["fetch_time"], "gzip_encoded_body": raw[:2] == b"\x1f\x8b",
            "decoded_bytes": len(dec), "decoded_sha256": hashlib.sha256(dec).hexdigest(),
            "identical_bytes_to_live_copy": dec == live_raw,
            "sentence_present": SENT in joined, "date_present": DATE in joined, "first_date_string": m.group(1) if m else None,
            "sentence_line": [l for l in lines if "GPT-6 Astra" in l],
            "text_diff_vs_live": {"removed_lines": sum(1 for d in diff if d.startswith("-") and not d.startswith("---")),
                                  "added_lines": sum(1 for d in diff if d.startswith("+") and not d.startswith("+++")),
                                  "file": os.path.basename(dfn)},
            "decoded_file": os.path.basename(fn)}
    res["text_provenance"] = F["text_provenance"]
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("sentence_line",)} for k, v in res["snapshots"].items()}, indent=1))
    print(res["live_copy"]["sentence_present"], res["live_copy"]["date_present"])


if __name__ == "__main__":
    main()
