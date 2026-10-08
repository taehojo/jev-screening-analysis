"""AN-0002-04 step 2: second-round retrieval that depends on step 1, and text extraction.

- arXiv API returned HTTP 406 in step 1; the title and metadata are read from the saved abstract page
  (citation_* meta tags). Crossref and DBLP are then searched with that title for a published version.
- For the three statistics papers, PubMed records (efetch, abstracts) are retrieved for the PMIDs returned
  by the step 1 title searches; for Fagerland 2013 the PMC full text is retrieved if a PMCID is listed.
- Cormack and Grossman 2016: the open access location reported by OpenAlex (if any) is fetched.
- Texts are extracted (PyMuPDF for PDF, lxml for XML and HTML) into sources/*.txt.
Queries contain only titles and identifiers of published papers.
"""
import json
import os
import sys
import time
import urllib.parse

import fitz  # PyMuPDF
import lxml.etree as ET
import lxml.html

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from an04_fetch_sources import SRC, fetch, q  # noqa: E402


def txt(name, lines):
    open(os.path.join(SRC, name), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    # arXiv metadata from the abstract page
    doc = lxml.html.fromstring(open(os.path.join(SRC, "arxiv_2106.09871_abs.html"), "rb").read())
    meta = {}
    for m in doc.xpath("//meta[starts-with(@name,'citation_')]"):
        meta.setdefault(m.get("name"), []).append(m.get("content"))
    sub = doc.xpath("//div[@class='submission-history']")
    meta["submission_history"] = [" ".join(sub[0].text_content().split())] if sub else []
    json.dump(meta, open(os.path.join(SRC, "arxiv_2106.09871_abs_meta.json"), "w"), indent=1, ensure_ascii=False)
    title = meta.get("citation_title", [None])[0]
    print("arXiv title:", title)
    # full text of the PDF
    pdf = fitz.open(os.path.join(SRC, "arxiv_2106.09871.pdf"))
    lines = []
    for i, p in enumerate(pdf, 1):
        lines.append(f"=== page {i} ===")
        lines.extend(p.get_text().splitlines())
    txt("arxiv_2106.09871.txt", lines)
    if title:
        fetch("crossref_search_yang_lewis_frieder.json",
              "https://api.crossref.org/works?rows=5&query.bibliographic=" + q(title) + "&query.author=" + q("Yang Lewis Frieder"))
        fetch("dblp_search_yang_lewis_frieder.json", "https://dblp.org/search/publ/api?format=json&h=10&q=" + q(title))
    # Repke et al. full text
    for fn in ("europepmc_PMC12825451_fulltext.xml", "ncbi_pmc_12825451_efetch.xml"):
        root = ET.parse(os.path.join(SRC, fn)).getroot()
        out = []
        for el in root.iter():
            if el.tag in ("title", "p", "article-title", "caption", "td", "th", "label"):
                s = " ".join("".join(el.itertext()).split())
                if s:
                    out.append(f"<{el.tag}> {s}")
        txt(fn.replace(".xml", ".txt"), out)
    # PubMed records for the statistics papers
    for key in ("fagerland_2013", "tango_1998", "newcombe_1998"):
        d = json.load(open(os.path.join(SRC, f"pubmed_esearch_{key}.json")))
        ids = d["esearchresult"]["idlist"]
        print(key, "PMIDs", ids)
        if ids:
            time.sleep(1)
            fetch(f"pubmed_efetch_{key}.xml",
                  "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id=" + ",".join(ids))
    # PMC full text of Fagerland 2013 if a PMCID is listed in the PubMed record
    try:
        root = ET.parse(os.path.join(SRC, "pubmed_efetch_fagerland_2013.xml")).getroot()
        pmc = [e.text for e in root.iter("ArticleId") if e.get("IdType") == "pmc"]
        print("Fagerland PMCID", pmc)
        if pmc:
            time.sleep(1)
            fetch("europepmc_fagerland_2013_fulltext.xml",
                  f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc[0]}/fullTextXML")
            r2 = ET.parse(os.path.join(SRC, "europepmc_fagerland_2013_fulltext.xml")).getroot()
            out = []
            for el in r2.iter():
                if el.tag in ("title", "p", "article-title"):
                    s = " ".join("".join(el.itertext()).split())
                    if s:
                        out.append(f"<{el.tag}> {s}")
            txt("europepmc_fagerland_2013_fulltext.txt", out)
    except Exception as ex:
        print("Fagerland full text step failed:", ex)
    # Cormack and Grossman: open access location from OpenAlex
    oa = json.load(open(os.path.join(SRC, "openalex_search_cormack_grossman_2016.json")))
    for w in oa.get("results", [])[:5]:
        print("OpenAlex:", w.get("display_name"), w.get("doi"), w.get("publication_year"),
              (w.get("open_access") or {}).get("oa_status"), (w.get("open_access") or {}).get("oa_url"))
    first = oa["results"][0] if oa.get("results") else None
    if first and (first.get("open_access") or {}).get("oa_url"):
        u = first["open_access"]["oa_url"]
        fetch("cormack_grossman_2016_oa_copy", u)


if __name__ == "__main__":
    main()
