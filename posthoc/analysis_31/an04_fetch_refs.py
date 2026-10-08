"""AN-0003-04 (round 3, literature check, I-8): bibliographic data of references 24 and 14 from the sources.

Plan: REVISION_ANALYSIS_LOG.md, entry of 2026-09-28 10:28:33 EDT (AN-0003-04) and the supplement of
2026-09-28 10:34:49 EDT.

GET requests only, to api.crossref.org, eutils.ncbi.nlm.nih.gov, dl.acm.org and doi.org. The queries contain only the
identifiers already printed in the manuscript (DOI 10.1145/3469096.3469873 from AN-0002-04, DOI
10.1016/j.jclinepi.2026.112514 of reference 14, PMID 42767575 from AN-0001-18). No manuscript text, study record or
label is sent. Every response is saved in sources/ with HTTP status and time (sources/fetch_record.tsv).
Standard library urllib (requests is not installed in review_pipeline/.venv_tools; no package is installed).
"""
import os
import subprocess
import time
import urllib.error
import urllib.request

OUT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(OUT, "sources")
os.makedirs(SRC, exist_ok=True)
UA = "Mozilla/5.0 (literature verification for a manuscript revision; mailto not given)"
REC = os.path.join(SRC, "fetch_record.tsv")


def now_et():
    return subprocess.run(["date", "+%Y-%m-%d %H:%M:%S %Z"], capture_output=True, text=True).stdout.strip()


def fetch(name, url, accept=None, tries=4):
    wait = 5
    status = None
    for k in range(tries):
        t = now_et()
        status, body, ctype, final = None, b"", "", url
        try:
            h = {"User-Agent": UA}
            if accept:
                h["Accept"] = accept
            req = urllib.request.Request(url, headers=h)
            with urllib.request.urlopen(req, timeout=90) as r:
                body = r.read(); status = r.status; ctype = r.headers.get("Content-Type", ""); final = r.geturl()
        except urllib.error.HTTPError as e:
            status = e.code
            try:
                body = e.read()
            except Exception:
                body = b""
        except Exception as e:
            status = f"error:{type(e).__name__}"
        with open(REC, "a") as f:
            f.write(f"{t}\t{name}\t{status}\t{len(body)}\t{ctype}\t{url}\t{final}\n")
        print(t, name, status, len(body), url, "->", final, flush=True)
        if status == 200:
            open(os.path.join(SRC, name), "wb").write(body)
            return status
        if body:
            open(os.path.join(SRC, f"{name}.http{status}"), "wb").write(body)
        if status in (429, 500, 502, 503, 504) or str(status).startswith("error"):
            time.sleep(wait); wait *= 2
            continue
        return status
    return status


if __name__ == "__main__":
    with open(REC, "a") as f:
        f.write("time\tname\tstatus\tbytes\tcontent_type\turl\tfinal_url\n")
    fetch("crossref_10.1145_3469096.3469873.json", "https://api.crossref.org/works/10.1145/3469096.3469873")
    time.sleep(1)
    fetch("crossref_10.1016_j.jclinepi.2026.112514.json", "https://api.crossref.org/works/10.1016/j.jclinepi.2026.112514")
    time.sleep(1)
    fetch("pubmed_efetch_42767575.xml", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=42767575&retmode=xml")
    time.sleep(1)
    fetch("acm_dl_doi_10.1145_3469096.3469873.html", "https://dl.acm.org/doi/10.1145/3469096.3469873", tries=2)
    time.sleep(1)
    fetch("doi_org_10.1016_j.jclinepi.2026.112514.html", "https://doi.org/10.1016/j.jclinepi.2026.112514", tries=2)
    time.sleep(1)
    # Crossref content negotiation through doi.org (citeproc JSON), a second route to the registered metadata
    fetch("doi_org_citeproc_10.1145_3469096.3469873.json", "https://doi.org/10.1145/3469096.3469873", accept="application/vnd.citationstyles.csl+json", tries=2)
    time.sleep(1)
    fetch("doi_org_citeproc_10.1016_j.jclinepi.2026.112514.json", "https://doi.org/10.1016/j.jclinepi.2026.112514", accept="application/vnd.citationstyles.csl+json", tries=2)
    print("done", now_et(), flush=True)
