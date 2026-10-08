"""AN-0003-04 step 2: extract the bibliographic fields from the saved responses and compare them with the current
entries of references 24 and 14 (v0002 draft). No network. Writes results.json."""
import datetime
import json
import os
import re

from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "sources")
DRAFT = "/N/project/AiLab/jev/review_pipeline/versions/v0002_after_round0002/source/draft_v2.md"
AN0204 = "/N/project/AiLab/jev/review_pipeline/rounds/round_0002/revision/analysis/AN-0002-04/sources"


def entry(n):
    for line in open(DRAFT, encoding="utf-8"):
        if line.startswith(f"{n}. "):
            return line.strip()


def initials(given):
    return "".join(p[0] for p in re.split(r"[\s.\-]+", given) if p)


def cr(path):
    m = json.load(open(path))["message"]
    return {"DOI": m.get("DOI"), "type": m.get("type"), "title": (m.get("title") or [None])[0],
            "container_title": (m.get("container-title") or [None])[0], "page": m.get("page"), "article_number": m.get("article-number"),
            "volume": m.get("volume"), "issue": m.get("issue"), "issued": m.get("issued"), "published_online": m.get("published-online"),
            "published_print": m.get("published-print"), "publisher": m.get("publisher"), "event": m.get("event"),
            "authors": [{"given": a.get("given"), "family": a.get("family"), "initials": initials(a.get("given", ""))} for a in m.get("author", [])]}


def csl(path):
    m = json.load(open(path))
    return {"page": m.get("page"), "article_number": m.get("article-number"), "volume": m.get("volume"), "container_title": m.get("container-title"),
            "issued": m.get("issued"), "authors": [{"given": a.get("given"), "family": a.get("family")} for a in m.get("author", [])]}


def pubmed(path):
    t = etree.parse(path)
    return {"pmid": t.findtext(".//PMID"), "title": t.findtext(".//ArticleTitle"), "journal": t.findtext(".//Journal/Title"),
            "iso_abbreviation": t.findtext(".//ISOAbbreviation"), "volume": t.findtext(".//JournalIssue/Volume"), "issue": t.findtext(".//JournalIssue/Issue"),
            "pub_date": {c.tag: c.text for c in t.find(".//JournalIssue/PubDate")},
            "medline_pgn": t.findtext(".//Pagination/MedlinePgn"),
            "elocation": {x.get("EIdType"): x.text for x in t.findall(".//ELocationID")},
            "article_date_electronic": {c.tag: c.text for c in t.find(".//ArticleDate")} if t.find(".//ArticleDate") is not None else None,
            "publication_status": t.findtext(".//PublicationStatus"),
            "author_list_complete": t.find(".//AuthorList").get("CompleteYN"),
            "authors": [{"last": a.findtext("LastName"), "initials": a.findtext("Initials"), "fore": a.findtext("ForeName")} for a in t.findall(".//AuthorList/Author")]}


def main():
    fetch = [l.rstrip("\n").split("\t") for l in open(os.path.join(SRC, "fetch_record.tsv")) if not l.startswith("time\t")]
    r24 = {"current_entry_v0002": entry(24),
           "crossref": cr(os.path.join(SRC, "crossref_10.1145_3469096.3469873.json")),
           "doi_org_csl": csl(os.path.join(SRC, "doi_org_citeproc_10.1145_3469096.3469873.json")),
           "acm_dl_page": "HTTP 403 (sources/acm_dl_doi_10.1145_3469096.3469873.html.http403, a bot-check page)",
           "arxiv_acm_reference_format": None}
    txt = open(os.path.join(AN0204, "arxiv_2106.09871.txt"), encoding="utf-8").read()
    m = re.search(r"ACM Reference Format:\s*(.+?https://doi\.org/\S+)", txt, flags=re.S)
    r24["arxiv_acm_reference_format"] = re.sub(r"\s+", " ", m.group(1)) if m else None
    c = r24["crossref"]
    r24["fields"] = {
        "authors": {"verified": [f"{a['family']} {a['initials']}" for a in c["authors"]] == ["Yang E", "Lewis DD", "Frieder O"],
                    "value": ", ".join(f"{a['family']} {a['initials']}" for a in c["authors"]), "source": "Crossref"},
        "title": {"verified": c["title"].lower() == "heuristic stopping rules for technology-assisted review", "value": c["title"], "source": "Crossref"},
        "container": {"verified": c["container_title"] == "Proceedings of the 21st ACM Symposium on Document Engineering", "value": c["container_title"], "source": "Crossref"},
        "year": {"verified": c["issued"]["date-parts"][0][0] == 2021, "value": c["issued"]["date-parts"][0], "source": "Crossref"},
        "pages": {"verified": c["page"] == "1-10" and r24["doi_org_csl"]["page"] == "1-10", "value": c["page"], "source": "Crossref and doi.org CSL"},
        "article_number": {"verified": False, "value": c["article_number"],
                           "note": "not in the Crossref or CSL record; the ACM Digital Library page returned HTTP 403; the ACM reference format printed in the arXiv PDF gives '10 pages' and no article number"},
        "doi": {"verified": c["DOI"] == "10.1145/3469096.3469873", "value": c["DOI"], "source": "Crossref (registered DOI resolves)"}}
    r14 = {"current_entry_v0002": entry(14),
           "crossref": cr(os.path.join(SRC, "crossref_10.1016_j.jclinepi.2026.112514.json")),
           "doi_org_csl": csl(os.path.join(SRC, "doi_org_citeproc_10.1016_j.jclinepi.2026.112514.json")),
           "pubmed": pubmed(os.path.join(SRC, "pubmed_efetch_42767575.xml")),
           "publisher_page": "doi.org redirected to linkinghub.elsevier.com (HTTP 200, redirect page to ScienceDirect PII S0895435626003902); the article page itself was not fetched"}
    c, p = r14["crossref"], r14["pubmed"]
    cr_auth = [f"{a['family']} {a['initials']}" for a in c["authors"]]
    pm_auth = [f"{a['last']} {a['initials']}" for a in p["authors"]]
    r14["fields"] = {
        "authors": {"verified": cr_auth == pm_auth and len(pm_auth) == 6 and p["author_list_complete"] == "Y", "value": ", ".join(pm_auth),
                    "crossref": ", ".join(cr_auth), "pubmed": ", ".join(pm_auth), "source": "PubMed (initials as indexed) and Crossref (given names)"},
        "title": {"verified": p["title"].rstrip(".") == c["title"], "value": c["title"], "source": "Crossref and PubMed"},
        "journal": {"verified": p["iso_abbreviation"] == "J Clin Epidemiol", "value": p["iso_abbreviation"], "source": "PubMed"},
        "article_number": {"verified": c["article_number"] == "112514" and p["medline_pgn"] == "112514", "value": "112514", "source": "Crossref article-number and PubMed MedlinePgn"},
        "volume": {"verified": c["volume"] is None and p["volume"] is None, "value": None, "note": "no volume assigned in either record (ahead of print)"},
        "online_date": {"verified": p["article_date_electronic"] == {"Year": "2026", "Month": "09", "Day": "21"}, "value": "2026-09-21", "source": "PubMed ArticleDate (Electronic)"},
        "status": {"value": p["publication_status"], "source": "PubMed"},
        "doi": {"verified": c["DOI"] == "10.1016/j.jclinepi.2026.112514" and p["elocation"].get("doi") == "10.1016/j.jclinepi.2026.112514", "value": c["DOI"], "source": "Crossref and PubMed"},
        "crossref_issued": {"value": c["issued"], "note": "Crossref gives year and month only (2026-09)"}}
    out = {"analysis_id": "AN-0003-04", "type": "literature check", "fetch_record": fetch, "reference_24": r24, "reference_14": r14,
           "finished": datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")}
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False)
    for n, r in (("24", r24), ("14", r14)):
        print("== reference", n)
        print("   current:", r["current_entry_v0002"])
        for k, v in r["fields"].items():
            print("  ", k, v)
    print("   arXiv ACM reference format:", r24["arxiv_acm_reference_format"])


if __name__ == "__main__":
    main()
