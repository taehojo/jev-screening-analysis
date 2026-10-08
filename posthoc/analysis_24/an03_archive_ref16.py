"""AN-0002-03 (round 2, post hoc documentary check).

Reference 16: re-query the Internet Archive CDX index of the vendor blog post
(read only), fetch with the id_ flag the latest snapshot taken at or before the
access time of the live copy (2026-09-26 19:00:40 EDT = 2026-09-26 23:00:40 UTC)
and the earliest snapshot, save them with HTTP status and fetch time, extract
text, check the reference-model sentence and the publication date, and diff the
text against the live copy saved in round 1 (AN-0001-15/sources).

Save Page Now is not used. No manuscript content, no study record and no label is
sent anywhere; the only requests are GET requests to web.archive.org.

Part B lists, from the existing output audit_fix_iter4/fx41_text_provenance.json,
the first-appearance time and model of each text sent to the evaluated models.
"""
import datetime as dt
import difflib
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import lxml.html

OUT = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.join(OUT, "snapshots")
os.makedirs(SNAP, exist_ok=True)
R1 = "/N/project/AiLab/jev/review_pipeline/rounds/round_0001/revision/analysis"
LIVE_HTML = f"{R1}/AN-0001-15/sources/typesafe_blog_introducing_system_one_models_and_jev.html"
LIVE_TXT = f"{R1}/AN-0001-15/sources/typesafe_blog_introducing_system_one_models_and_jev.txt"
OLD_CDX = f"{R1}/AN-0001-15/sources/wayback_cdx_typesafe_ai_blog_introducing-system-one-models-and-jev.json"
PROV = f"{R1}/audit_fix_iter4/fx41_text_provenance.json"
PAGE = "typesafe.ai/blog/introducing-system-one-models-and-jev"
LIVE_URL = "https://" + PAGE
ACCESS_UTC = "20260926230040"  # live copy fetched Sat Sep 26 19:00:40 EDT 2026 (fetch_record.txt)
SENTENCE_KEYS = ["GPT-6 Astra", "Fable 5.1", "reference answer"]
FULL_SENTENCE = ("We use the average of GPT-6 Astra and Fable 5.1 as the reference answer")
DATE_STR = "Sep 15, 2026"
UA = "Mozilla/5.0 (research reference check; contact via journal)"
BACKOFF = [0, 15, 30, 60, 120, 240, 480]  # seconds, about 16 minutes in total

log_lines = []


def now_et():
    return subprocess.run(["date", "+%Y-%m-%d %H:%M:%S %Z"], capture_output=True, text=True).stdout.strip()


def log(msg):
    line = f"[{now_et()}] {msg}"
    log_lines.append(line)
    print(line, flush=True)


def get(url):
    """GET with exponential backoff on 429/5xx/network errors. Returns (status, body, time, attempts)."""
    attempts = []
    for wait in BACKOFF:
        if wait:
            log(f"  waiting {wait} s before retry")
            time.sleep(wait)
        t = now_et()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
                attempts.append({"time": t, "status": r.status})
                log(f"GET {url} -> {r.status} ({len(body)} bytes)")
                return r.status, body, t, attempts
        except urllib.error.HTTPError as e:
            attempts.append({"time": t, "status": e.code})
            log(f"GET {url} -> HTTP {e.code}")
            if e.code in (404, 403):
                return e.code, e.read() if hasattr(e, "read") else b"", t, attempts
        except Exception as e:  # network error
            attempts.append({"time": t, "status": f"error: {type(e).__name__}: {e}"})
            log(f"GET {url} -> error {type(e).__name__}: {e}")
    return None, b"", now_et(), attempts


def html_text(raw):
    doc = lxml.html.fromstring(raw)
    for bad in doc.xpath("//script|//style|//noscript"):
        bad.getparent().remove(bad)
    txt = doc.text_content()
    lines = [" ".join(l.split()) for l in txt.splitlines()]
    return [l for l in lines if l]


def main():
    res = {"analysis_id": "AN-0002-03", "started": now_et(), "page": LIVE_URL,
           "access_time_live_copy_utc": ACCESS_UTC,
           "note": "post hoc documentary check; no model call; no manuscript content or study record sent"}
    # Part A1: CDX index
    cdx_url = f"https://web.archive.org/cdx/search/cdx?url={PAGE}&output=json&fl=timestamp,statuscode,digest,mimetype,original"
    st, body, t, att = get(cdx_url)
    res["cdx_query"] = {"url": cdx_url, "status": st, "time": t, "attempts": att}
    rows = None
    if st == 200 and body:
        open(os.path.join(OUT, "cdx_blog_post_requery.json"), "wb").write(body)
        data = json.loads(body.decode("utf-8"))
        rows = [dict(zip(data[0], r)) for r in data[1:]]
        res["cdx_source"] = "re-queried index (cdx_blog_post_requery.json)"
    else:
        old = json.load(open(OLD_CDX))
        rows = [dict(zip(old[0], r)) for r in old[1:]]
        for r in rows:
            r.setdefault("original", LIVE_URL)
        res["cdx_source"] = "re-query failed; saved round 1 index (2026-09-26 19:08 EDT) used"
    ok = [r for r in rows if r.get("statuscode") == "200"]
    res["cdx_n_rows"] = len(rows)
    res["cdx_n_status200"] = len(ok)
    res["cdx_first"] = rows[0]["timestamp"] if rows else None
    res["cdx_last"] = rows[-1]["timestamp"] if rows else None
    res["cdx_n_distinct_digests_status200"] = len({r["digest"] for r in ok})
    before = [r for r in ok if r["timestamp"] <= ACCESS_UTC]
    targets = {}
    if ok:
        targets["earliest"] = ok[0]
    if before:
        targets["latest_at_or_before_access"] = before[-1]
    # Part A2: fetch snapshots with id_
    live_lines = html_text(open(LIVE_HTML, "rb").read())
    live_saved_txt = [l for l in (" ".join(x.split()) for x in open(LIVE_TXT, encoding="utf-8").read().splitlines()) if l]
    res["live_copy"] = {"html": LIVE_HTML, "sha256": hashlib.sha256(open(LIVE_HTML, "rb").read()).hexdigest(),
                        "n_text_lines": len(live_lines),
                        "sentence_present": FULL_SENTENCE in " ".join(live_lines),
                        "date_present": DATE_STR in " ".join(live_lines)}
    snaps = {}
    for label, r in targets.items():
        ts = r["timestamp"]
        orig = r.get("original", LIVE_URL)
        url = f"https://web.archive.org/web/{ts}id_/{orig}"
        st, body, t, att = get(url)
        rec = {"cdx_timestamp": ts, "cdx_digest": r["digest"], "snapshot_url": f"https://web.archive.org/web/{ts}/{orig}",
               "fetched_url": url, "http_status": st, "fetch_time": t, "attempts": att}
        if st == 200 and body:
            fn = os.path.join(SNAP, f"blog_{ts}_id.html")
            open(fn, "wb").write(body)
            lines = html_text(body)
            open(os.path.join(SNAP, f"blog_{ts}_id.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
            joined = " ".join(lines)
            rec.update({"saved_as": fn, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                        "n_text_lines": len(lines),
                        "sentence_present": FULL_SENTENCE in joined,
                        "sentence_keys_present": {k: (k in joined) for k in SENTENCE_KEYS},
                        "date_present": DATE_STR in joined,
                        "sentence_lines": [l for l in lines if "GPT-6 Astra" in l or "Fable" in l]})
            diff = list(difflib.unified_diff(live_lines, lines, fromfile="live_copy_2026-09-26_190040EDT",
                                             tofile=f"snapshot_{ts}", lineterm="", n=0))
            open(os.path.join(OUT, f"diff_live_vs_{ts}.txt"), "w", encoding="utf-8").write("\n".join(diff) + "\n")
            rec["diff_vs_live"] = {"n_removed_lines": sum(1 for d in diff if d.startswith("-") and not d.startswith("---")),
                                   "n_added_lines": sum(1 for d in diff if d.startswith("+") and not d.startswith("+++")),
                                   "file": f"diff_live_vs_{ts}.txt"}
        snaps[label] = rec
    res["snapshots"] = snaps
    # sanity: live html text versus the round 1 saved text file
    d2 = list(difflib.unified_diff(live_saved_txt, live_lines, lineterm="", n=0))
    res["live_html_text_vs_round1_txt_n_diff_lines"] = sum(1 for d in d2 if d[:1] in "+-" and d[:3] not in ("---", "+++"))
    # Part B: provenance of the texts sent to the models
    prov = json.load(open(PROV))
    out = {}
    for k, v in prov["results"].items():
        f = v.get("first_model_tool_call") or v.get("first_any")
        out[k] = {"n_entries": v.get("n_entries"),
                  "first_time_et": f.get("time_et") if f else None,
                  "first_model": f.get("model") if f else None,
                  "first_kind": f.get("kind") if f else None,
                  "first_any_time_et": (v.get("first_any") or {}).get("time_et"),
                  "first_any_model": (v.get("first_any") or {}).get("model")}
    res["text_provenance"] = {"source": PROV, "source_written": prov.get("written"), "texts": out}
    res["finished"] = now_et()
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, ensure_ascii=False)
    open(os.path.join(OUT, "an03_requests.log"), "w").write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
