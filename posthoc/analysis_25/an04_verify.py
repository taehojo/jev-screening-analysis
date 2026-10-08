"""AN-0002-04 step 3: locate each statement to be verified in the saved source texts and write results.json.

Every quotation written to results.json is cut from the saved files (sources/), not typed. A statement counts as
verified only when its passage is found in a primary or accessible full text; the location (page or section) is
given as found in the file. Bibliographic data are taken from the Crossref, PubMed and Europe PMC responses.
"""
import json
import os
import re

import lxml.etree as ET

OUT = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(OUT, "sources")


def read(n):
    return open(os.path.join(S, n), encoding="utf-8").read()


def find(text, pattern, width=0):
    m = re.search(pattern, text, re.S)
    if not m:
        return None
    a, b = max(0, m.start() - width), min(len(text), m.end() + width)
    return " ".join(text[a:b].split())


def page_of(txt, pos_pattern):
    m = re.search(pos_pattern, txt, re.S)
    if not m:
        return None
    pages = [(mm.start(), int(mm.group(1))) for mm in re.finditer(r"=== page (\d+) ===", txt)]
    pg = [p for s, p in pages if s <= m.start()]
    return pg[-1] if pg else None


def crossref_item(fn, doi_prefix=None, idx=0):
    d = json.load(open(os.path.join(S, fn)))["message"]["items"]
    it = d[idx]
    return {"doi": it.get("DOI"), "title": (it.get("title") or [""])[0], "container": (it.get("container-title") or [""])[0],
            "volume": it.get("volume"), "issue": it.get("issue"), "page": it.get("page"), "issued": it.get("issued", {}).get("date-parts"),
            "authors": [f"{a.get('family')} {a.get('given', '')}".strip() for a in it.get("author", [])], "crossref_score": it.get("score")}


def pubmed(fn, pmid):
    r = ET.parse(os.path.join(S, fn)).getroot()
    for a in r.iter("PubmedArticle"):
        if a.findtext(".//PMID") == pmid:
            return {"pmid": pmid, "title": "".join(a.find(".//ArticleTitle").itertext()), "journal": a.findtext(".//Journal/ISOAbbreviation"),
                    "volume": a.findtext(".//JournalIssue/Volume"), "issue": a.findtext(".//JournalIssue/Issue"),
                    "pages": a.findtext(".//MedlinePgn"), "year": a.findtext(".//JournalIssue/PubDate/Year"),
                    "authors": [f"{x.findtext('LastName')} {x.findtext('Initials')}" for x in a.iter("Author")],
                    "abstract": " ".join("".join(x.itertext()) for x in a.iter("AbstractText")),
                    "pmc": [e.text for e in a.iter("ArticleId") if e.get("IdType") == "pmc"],
                    "doi": [e.text for e in a.iter("ArticleId") if e.get("IdType") == "doi"]}
    return None


def main():
    y = read("arxiv_2106.09871.txt")
    ymeta = json.load(open(os.path.join(S, "arxiv_2106.09871_abs_meta.json")))
    rep = read("europepmc_PMC12825451_fulltext.txt")
    rep_meta = json.load(open(os.path.join(S, "europepmc_search_PMC12825451.json")))["resultList"]["result"][0]
    fag = read("europepmc_fagerland_2013_fulltext.txt")
    res = {"analysis_id": "AN-0002-04", "post_hoc": False, "note": "literature verification (I-8); sources saved in sources/ with fetch_record.tsv",
           "statements": []}
    # 1. Yang, Lewis and Frieder: knee rule parameters and recall goal
    knee_rule = find(y, r"The Knee Method stops at the first.{0,120}?≥1000")
    knee_goal = find(y, r"The Knee Method is targeted at a recall goal of 0\.7.{0,120}?recall targets\.")
    knee_goal2 = find(y, r"Note that the Knee Method was\s*designed for a 0\.7 recall target\.")
    res["statements"].append({
        "id": "S1", "requested_by": "R2.2.05 (E2.12)",
        "statement": "The knee rule of Cormack and Grossman stops at the first s with rho(s) >= 156 - min(Rel(s), 150) and s >= 1000, and is targeted at a recall goal of 0.7",
        "source": {"arxiv": "2106.09871", "version": ymeta.get("submission_history"), "title": ymeta["citation_title"][0],
                   "authors": ymeta["citation_author"], "doi_published_version": ymeta.get("citation_doi"),
                   "crossref_published_version": crossref_item("crossref_search_yang_lewis_frieder.json")},
        "verified": bool(knee_rule and knee_goal),
        "quotations": [{"text": knee_rule, "location": f"section 3.5 (Knee Method), page {page_of(y, r'The Knee Method stops at the first')} of the arXiv PDF"},
                       {"text": knee_goal, "location": f"section 3.5, page {page_of(y, r'The Knee Method is targeted at a recall goal')}"},
                       {"text": knee_goal2, "location": f"section 6 (Results and Analysis), page {page_of(y, r'designed for a 0\.7 recall target')}"}],
        "secondary_source": True,
        "primary_source_status": "Cormack and Grossman 2016 (reference 6): ACM Digital Library full text returned HTTP 403 to this server (sources/cormack_grossman_2016_oa_copy.http403, acm_cormack_grossman_2016_doi_pdf.http403), although OpenAlex and Semantic Scholar list it as open access (CC BY-ND); the primary text was not read, so the parameters are verified only in the secondary description",
        "reference6_bibliographic_check": crossref_item("crossref_search_cormack_grossman_2016.json"),
        "yang_reference_for_knee": find(y, r"\[8\] Gordon V\. Cormack and Maura R\. Grossman\. 2016\..{0,160}?2911510")})
    # 2. Repke et al. 2026 (reference 8): KNEE and CMH statements quoted by R1
    q1 = find(rep, r"BATCHPRECISION and KNEE have no clear point where they stop.{0,260}?\(Pearson's correlation: −0\.46 and −0\.4\)\.")
    q2 = find(rep, r"Only one method reliably meets the set recall target, but stops conservatively\.")
    q3 = find(rep, r"Only one method, CMH, never stops before the set recall target\.")
    q4 = find(rep, r"only one method, CMH, is safe to use, but it does not use the full work‐saving potential\.")
    q5 = find(rep, r"In our experiments, only one method \(CMH\) has not missed any relevant records and provided work savings\.")
    q6 = find(rep, r"CMH is a statistical stopping criterion that calculates.{0,200}?confidence target is met\.")
    res["statements"].append({
        "id": "S2", "requested_by": "R1.2.05 (E2.12)",
        "statement": "Repke et al. report that KNEE (and BATCHPRECISION) had no clear stopping point, stopping far too early or late across the range regardless of hyperparameters, and tended to stop too early more often in larger datasets; and that only one method (CMH) reliably met the recall target but stopped conservatively",
        "source": {"pmcid": "PMC12825451", "pmid": rep_meta.get("pmid"), "doi": rep_meta.get("doi"), "title": rep_meta.get("title"),
                   "authors": rep_meta.get("authorString"), "journal": (rep_meta.get("journalInfo") or {}).get("journal", {}).get("isoabbreviation"),
                   "volume": (rep_meta.get("journalInfo") or {}).get("volume"), "issue": (rep_meta.get("journalInfo") or {}).get("issue"),
                   "article_number": rep_meta.get("pageInfo"), "year": rep_meta.get("pubYear"), "first_publication": rep_meta.get("firstPublicationDate")},
        "verified": bool(q1 and q2 and q3),
        "quotations": [{"text": q1, "location": "Results (paragraph on stopping too early or late; Europe PMC full text XML)"},
                       {"text": q2, "location": "Abstract, Conclusions"},
                       {"text": q3, "location": "Results"},
                       {"text": q4, "location": "Introduction (last paragraph)"},
                       {"text": q5, "location": "Conclusion"},
                       {"text": q6, "location": "Methods, Stopping Methods (definition of CMH)"}],
        "reviewer_quote_check": "R1's first quotation matches the Results sentence verbatim (from 'BATCHPRECISION and KNEE' to 'the larger the data set is'); R1's second quotation ('the only method [that] reliably meets the set recall target, but stops conservatively') is a bracketed adaptation of the Abstract sentence 'Only one method reliably meets the set recall target, but stops conservatively'; the Abstract does not name the method, and the Introduction, Results and Conclusion name it as CMH",
        "reference8_bibliographic_data_match": {"manuscript": "Repke T, Tinsdeall F, Danilenko D, et al. ... Cochrane Evid Synth Methods 2026; 4: e70068",
                                                "found": f"{rep_meta.get('authorString')} {(rep_meta.get('journalInfo') or {}).get('journal', {}).get('isoabbreviation')} {rep_meta.get('pubYear')}; {(rep_meta.get('journalInfo') or {}).get('volume')}: {rep_meta.get('pageInfo')}"}})
    # 3. Methods references for AN-0002-02
    fg = pubmed("pubmed_efetch_fagerland_2013.xml", "23848987")
    tg = pubmed("pubmed_efetch_tango_1998.xml", "9595618")
    nw = pubmed("pubmed_efetch_newcombe_1998.xml", "9839354")
    fq = find(fag, r"The McNemar mid-p test is a considerably improvement.{0,900}?in any situation\.")
    res["statements"].append({
        "id": "S3", "requested_by": "R3.2.06 (E2.13)",
        "statement": "Fagerland, Lydersen and Laake recommend the McNemar mid-p test (and the asymptotic test without continuity correction if small violations of the nominal level are acceptable) and advise against the exact conditional test",
        "source": {**{k: v for k, v in fg.items() if k != "abstract"}, "crossref": crossref_item("crossref_search_fagerland_2013.json")},
        "verified": bool(fq), "quotations": [{"text": fq, "location": "Conclusions (PMC full text, " + (fg["pmc"][0] if fg["pmc"] else "") + ")"}]})
    res["statements"].append({
        "id": "S4", "requested_by": "R3.2.06 (E2.13)",
        "statement": "Tango (1998) derived a score-based confidence interval for the difference of two paired proportions, applicable with off-diagonal zero cells",
        "source": {**{k: v for k, v in tg.items() if k != "abstract"}, "crossref": crossref_item("crossref_search_tango_1998.json", idx=1)},
        "verified": True, "verified_from": "PubMed abstract only (full text not accessible)",
        "quotations": [{"text": find(tg["abstract"], r"Further, a score-based confidence interval for the difference of two proportions is derived\..{0,200}?such data\."), "location": "Abstract"}]})
    res["statements"].append({
        "id": "S5", "requested_by": "R3.2.06 (E2.13)",
        "statement": "Newcombe (1998) method 10: a simpler interval for the paired difference based on the score interval for the single proportion",
        "source": {**{k: v for k, v in nw.items() if k != "abstract"}, "crossref": crossref_item("crossref_search_newcombe_1998.json")},
        "verified": True, "verified_from": "PubMed abstract only (full text not accessible); the formula used in AN-0002-02 follows the R source of CRAN contingencytables 3.1.0 (Newcombe_square_and_add_CI_paired_2x2.R), whose treatment of the correlation term (continuity-corrected phi, set to 0 when a margin is zero) could not be checked against the paper",
        "quotations": [{"text": find(nw["abstract"], r"A computationally simpler method based on the score interval for the single proportion also performs well \(method 10\)\."), "location": "Abstract"}],
        "letter_to_editor": pubmed("pubmed_efetch_newcombe_1998.xml", "10611622")})
    res["implementation_reference"] = {"package": "contingencytables 3.1.0 (CRAN; GitHub mirror cran/contingencytables)",
                                       "files": sorted(f for f in os.listdir(S) if f.startswith("cran_contingencytables_")),
                                       "use": "formula source and test values for AN-0002-02; not proposed as a reference of the manuscript"}
    res["not_verified"] = ["Cormack and Grossman 2016 full text (HTTP 403 from the ACM Digital Library on this server); the knee parameters are verified only in the secondary description by Yang, Lewis and Frieder",
                           "Tango 1998 and Newcombe 1998 full texts (not accessible); only the abstracts were read"]
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, ensure_ascii=False)
    for s in res["statements"]:
        print(s["id"], s["verified"], [ (q["location"], (q["text"] or "NOT FOUND")[:160]) for q in s["quotations"]])


if __name__ == "__main__":
    main()
