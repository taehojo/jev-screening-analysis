"""AN-0002-04 step 1 (round 2, post hoc literature check, I-8): retrieve and save the sources.

Only public bibliographic services and publisher or preprint pages are queried with GET requests.
The queries contain titles or identifiers of published papers only; no manuscript text, no study record
and no label is sent. Every response is saved in sources/ with its HTTP status and fetch time
(sources/fetch_record.tsv). Identifiers such as DOIs are taken from the responses of the services, not
from memory; the queries are bibliographic (title words) or the identifiers already printed in the
manuscript (arXiv 2106.09871 from the reviewer's comment, PMC12825451 from reference 8 as cited by the
reviewer and the round 1 records).
"""
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

OUT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(OUT, "sources")
os.makedirs(SRC, exist_ok=True)
UA = "Mozilla/5.0 (literature verification for a manuscript revision)"
REC = os.path.join(SRC, "fetch_record.tsv")


def now_et():
    return subprocess.run(["date", "+%Y-%m-%d %H:%M:%S %Z"], capture_output=True, text=True).stdout.strip()


def fetch(name, url, accept=None, tries=4):
    wait = 5
    for k in range(tries):
        t = now_et()
        status, body, ctype = None, b"", ""
        try:
            h = {"User-Agent": UA}
            if accept:
                h["Accept"] = accept
            req = urllib.request.Request(url, headers=h)
            with urllib.request.urlopen(req, timeout=90) as r:
                body = r.read()
                status = r.status
                ctype = r.headers.get("Content-Type", "")
        except urllib.error.HTTPError as e:
            status = e.code
            try:
                body = e.read()
            except Exception:
                body = b""
        except Exception as e:
            status = f"error:{type(e).__name__}"
        with open(REC, "a") as f:
            f.write(f"{t}\t{name}\t{status}\t{len(body)}\t{ctype}\t{url}\n")
        print(t, name, status, len(body), url, flush=True)
        if status == 200:
            open(os.path.join(SRC, name), "wb").write(body)
            return status, body
        if status in (429, 500, 502, 503, 504) or str(status).startswith("error"):
            time.sleep(wait)
            wait *= 2
            continue
        if body:
            open(os.path.join(SRC, name + f".http{status}"), "wb").write(body)
        return status, body
    return status, body


def q(s):
    return urllib.parse.quote(s)


def main():
    if not os.path.exists(REC):
        open(REC, "w").write("time\tsaved_as\thttp_status\tbytes\tcontent_type\turl\n")
    # 1. Yang, Lewis and Frieder: arXiv metadata and full text (current version)
    fetch("arxiv_2106.09871_api.xml", "http://export.arxiv.org/api/query?id_list=2106.09871")
    time.sleep(3)
    fetch("arxiv_2106.09871.pdf", "https://arxiv.org/pdf/2106.09871")
    time.sleep(3)
    fetch("arxiv_2106.09871_abs.html", "https://arxiv.org/abs/2106.09871")
    # published version: search by title taken from the arXiv response
    title = None
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(os.path.join(SRC, "arxiv_2106.09871_api.xml")).getroot()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        e = root.find("a:entry", ns)
        title = " ".join(e.find("a:title", ns).text.split())
    except Exception as ex:
        print("arXiv title not parsed:", ex)
    if title:
        fetch("crossref_search_yang_lewis_frieder.json",
              "https://api.crossref.org/works?rows=5&query.bibliographic=" + q(title) + "&query.author=" + q("Yang Lewis Frieder"))
        fetch("dblp_search_yang_lewis_frieder.json", "https://dblp.org/search/publ/api?format=json&h=10&q=" + q(title))
    # 2. Cormack and Grossman 2016 (reference 6)
    t6 = "Engineering quality and reliability in technology-assisted review"
    fetch("crossref_search_cormack_grossman_2016.json",
          "https://api.crossref.org/works?rows=5&query.bibliographic=" + q(t6) + "&query.author=" + q("Cormack Grossman"))
    fetch("openalex_search_cormack_grossman_2016.json",
          "https://api.openalex.org/works?per-page=5&search=" + q(t6))
    # 3. Repke et al. 2026 (reference 8), PMC12825451
    fetch("europepmc_PMC12825451_fulltext.xml",
          "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12825451/fullTextXML")
    time.sleep(1)
    fetch("ncbi_pmc_12825451_efetch.xml",
          "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=12825451&retmode=xml")
    time.sleep(1)
    fetch("europepmc_search_PMC12825451.json",
          "https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&resultType=core&query=" + q("PMCID:PMC12825451"))
    # 4. Methods references for AN-0002-02
    refs = {
        "fagerland_2013": ("The McNemar test for binary matched-pairs data: mid-p and asymptotic are better than exact conditional", "Fagerland Lydersen Laake"),
        "tango_1998": ("Equivalence test and confidence interval for the difference in proportions for the paired-sample design", "Tango"),
        "newcombe_1998": ("Improved confidence intervals for the difference between binomial proportions based on paired data", "Newcombe"),
    }
    for key, (t, a) in refs.items():
        fetch(f"crossref_search_{key}.json",
              "https://api.crossref.org/works?rows=3&query.bibliographic=" + q(t) + "&query.author=" + q(a))
        time.sleep(1)
        fetch(f"pubmed_esearch_{key}.json",
              "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&term=" + q(t + "[Title]"))
        time.sleep(1)


if __name__ == "__main__":
    main()
