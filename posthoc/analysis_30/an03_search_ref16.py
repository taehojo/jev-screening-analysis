"""AN-0003-03 (round 3, post hoc): does the vendor page of reference 16 contain a sentence stating that the vendor
cannot prove that its price is not subsidised? Search of our own saved copies.

Plan: REVISION_ANALYSIS_LOG.md, entry of 2026-09-28 10:28:33 EDT (AN-0003-03) and the supplement of
2026-09-28 10:34:49 EDT (search the saved text files and the tag-stripped HTML).

Step 1 (this script, no network): case-insensitive search for "subsidi", "sustainab", "pricing" and "price" in
  - the live copy saved 2026-09-26 19:00:40 EDT (AN-0001-15/sources), HTML and extracted text;
  - the Internet Archive captures 20260926091045 and 20260915192610 saved in AN-0002-03/snapshots (decoded HTML and
    extracted text).
Each match is written with the surrounding sentence, file, and capture time. Step 2 (network, only if the sentence is
absent from the capture of Sept 26) is in an03_cdx_later_captures.py.
"""
import datetime
import html
import json
import os
import re

from lxml import html as lh

HERE = os.path.dirname(os.path.abspath(__file__))
RP = "/N/project/AiLab/jev/review_pipeline/rounds"
S1 = RP + "/round_0001/revision/analysis/AN-0001-15/sources"
S2 = RP + "/round_0002/revision/analysis/AN-0002-03/snapshots"
COPIES = [
    ("live_20260926_190040_EDT", "live copy fetched 2026-09-26 19:00:40 EDT (https://typesafe.ai/blog/introducing-system-one-models-and-jev)",
     S1 + "/typesafe_blog_introducing_system_one_models_and_jev.html", S1 + "/typesafe_blog_introducing_system_one_models_and_jev.txt"),
    ("wayback_20260926091045", "Internet Archive capture 2026-09-26 09:10:45 UTC (id_)",
     S2 + "/blog_20260926091045_id_decoded.html", S2 + "/blog_20260926091045_id_decoded.txt"),
    ("wayback_20260915192610", "Internet Archive capture 2026-09-15 19:26:10 UTC (id_)",
     S2 + "/blog_20260915192610_id_decoded.html", S2 + "/blog_20260915192610_id_decoded.txt"),
]
TERMS = ["subsidi", "sustainab", "pricing", "price"]


def html_text(path):
    raw = open(path, "rb").read()
    doc = lh.fromstring(raw)
    for bad in doc.xpath("//script|//style|//noscript"):
        bad.getparent().remove(bad)
    t = doc.text_content()
    return re.sub(r"\s+", " ", t)


def raw_html_text(path):
    """Whole HTML including script payloads (framework pages carry text in JSON); tags removed, entities decoded."""
    s = open(path, encoding="utf-8", errors="replace").read()
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = s.replace("\\n", " ").replace('\\"', '"')
    return re.sub(r"\s+", " ", s)


def sentences_with(text, term):
    out = []
    for m in re.finditer(term, text, flags=re.I):
        a = max(text.rfind(". ", 0, m.start()), text.rfind("\n", 0, m.start()))
        b_candidates = [x for x in (text.find(". ", m.end()), text.find("\n", m.end())) if x != -1]
        b = min(b_candidates) + 1 if b_candidates else len(text)
        out.append(text[a + 1 if a >= 0 else 0:b].strip()[:600])
    # unique, in order
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq


def main():
    now = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    res = {"analysis_id": "AN-0003-03", "post_hoc": True, "step": 1, "started": now, "terms": TERMS, "copies": {}}
    for key, desc, hp, tp in COPIES:
        texts = {"extracted_text_file": re.sub(r"\s+", " ", open(tp, encoding="utf-8", errors="replace").read()),
                 "html_visible_text_lxml": html_text(hp),
                 "html_all_text_including_scripts": raw_html_text(hp)}
        C = {"description": desc, "html": hp, "text": tp, "matches": {}}
        for src, t in texts.items():
            C["matches"][src] = {term: sentences_with(t, term) for term in TERMS}
            C["matches"][src]["_counts"] = {term: len(re.findall(term, t, flags=re.I)) for term in TERMS}
        res["copies"][key] = C
        print("==", key, desc, flush=True)
        for src in texts:
            print("  ", src, C["matches"][src]["_counts"], flush=True)
            for term in TERMS:
                for s in C["matches"][src][term]:
                    print(f"     [{term}] {s}", flush=True)
    subs_26 = any(res["copies"]["wayback_20260926091045"]["matches"][s]["_counts"]["subsidi"] > 0 for s in res["copies"]["wayback_20260926091045"]["matches"])
    res["subsidi_in_capture_20260926091045"] = subs_26
    res["subsidi_in_any_saved_copy"] = any(res["copies"][k]["matches"][s]["_counts"]["subsidi"] > 0 for k in res["copies"] for s in res["copies"][k]["matches"])
    res["finished"] = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    json.dump(res, open(os.path.join(HERE, "results_step1.json"), "w"), indent=1, ensure_ascii=False)
    print("subsidi in capture 20260926091045:", subs_26, "; in any saved copy:", res["subsidi_in_any_saved_copy"], flush=True)


if __name__ == "__main__":
    main()
