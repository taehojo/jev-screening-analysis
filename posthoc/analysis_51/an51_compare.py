"""AN-0003-01 (manuscript analysis 51, post hoc): compare the definitions of Kusa, Lipani, Knoth & Hanbury
(Intell. Syst. Appl. 18, 200193, 2023; ref. kusa_wss) with the definitions of the Methods (Outcomes) of
v0002_after_round0002. No network. Inputs are read only.

Steps
1. Read the text extracted by an51_extract.py (sources/kusa_pdf_text.txt) and normalise it for searching
   (ligatures fi/ff/fl, line breaks and repeated spaces collapsed, soft line-number tokens are not removed,
   so every quotation below must be found verbatim in the normalised text; the script fails otherwise).
2. Locate each quotation and record its page.
3. Check exactly, with integer arithmetic, the identities used in the comparison:
   (a) n - floor(n*(1-0.95)) == ceil(0.95*n) for n = 1..100000 (Section 3.3 of the article against the
       ceil(0.95n)-th included record of the Methods);
   (b) ceil(0.95*n) == n exactly when n <= 19 (Section 3.5 against the Methods statement "at most 19").
   (c) the number of n in 1..1000 for which 0.05*n is not an integer (the cases in which the floor-less
       expressions of Section 4.1 are not integers).
4. Compare the bibliographic data of the PDF (cover page and metadata) with the Crossref record saved in
   round 1 (rounds/round_0001/revision/analysis/AN-0001-14/sources/crossref_10.1016_j.iswa.2023.200193.json).
5. Write results.json.
The judgements (agree / differ / not determinable) are made by the AI coding tool that wrote this script
(Claude Code) and are encoded below with the quotations they rest on; they are not an author's verification.
"""
import json, re, math
from fractions import Fraction

D = '/N/project/AiLab/jev/review_pipeline_npj/rounds/round_0003/revision/analysis/AN-0003-01'
CR = '/N/project/AiLab/jev/review_pipeline_npj/rounds/round_0001/revision/analysis/AN-0001-14/sources/crossref_10.1016_j.iswa.2023.200193.json'
MS = '/N/project/AiLab/jev/review_pipeline_npj/versions/v0002_after_round0002/source/manuscript_npjdm.md'
TN = '/N/project/AiLab/jev/review_pipeline_npj/versions/v0002_after_round0002/source/tables/tnr95.md'

raw = open(f'{D}/sources/kusa_pdf_text.txt').read()
pages = {}
for m in re.finditer(r'===== PAGE (\d+) =====\n(.*?)(?=\n\f===== PAGE |\Z)', raw, re.S):
    pages[int(m.group(1))] = m.group(2)

def norm(s):
    s = s.replace('ﬁ', 'fi').replace('ﬀ', 'ff').replace('ﬂ', 'fl').replace('ﬃ', 'ffi')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

npages = {k: norm(v) for k, v in pages.items()}

def locate(q):
    hits = [p for p, t in npages.items() if q in t]
    if not hits:
        raise SystemExit(f'QUOTATION NOT FOUND: {q!r}')
    return hits

# Quotations (normalised text). Printed page numbers of the pre-proof are the PDF page minus 2.
Q = {
 'cover_version': 'Journal Pre-proof',
 'cover_not_vor': 'it is not yet the definitive version of record',
 'highlight_tnr': 'Normalised WSS equals True Negative Rate (TNR).',
 'sec1_recommend': 'we recommend using TNR at r% recall as the evaluation measure for technology-assisted reviews.',
 'notation_nr': 'nr% rank of a document for which the recall level of r% is achieved',
 'sec33_heading': '3.3. The FN term',
 'sec33_floor': 'For a specific r% recall, the number of False Negatives (FN) is always equal to ⌊|I|·(1−r)⌋, where with ⌊·⌋we indicate the floor operator.',
 'sec33_tp': 'Consequently, for a fixed level of recall, true positives (TP) are equal to r · |I|.',
 'sec35_heading': '3.5. Evaluation with Cross-Validation',
 'sec35_zero_fn': 'such that for a specific level of recall r, (1 −r)% of relevant items would be fewer than one document (i.e., |I| · (1 −r) < 1), the number of false negatives will be equal to 0 for all recalls ≥r.',
 'sec35_eq6': 'WSS@r% = WSS@100% −(1 −r). (6)',
 'sec35_lt20': 'For WSS@95%, the equation above is true for all datasets where the total number of relevant documents used in the evaluation is fewer than 20 (|I| < 20).',
 'sec4_eq9_text': 'In the case of a recall threshold at 95%, the nWSS equation is: nWSS@95% = TN@95% |E| , (9)',
 'sec4_eq10_text': 'which is equal to the True Negative Rate (TNR), also known as specificity.',
 'sec4_eq11_text': 'nWSS@r% = TNR@r% = TN@r% |E| , (11)',
 'sec41_heading': '4.1. Alternative demonstration for rank-based evaluation',
 'sec41_nr': 'We assume that nr% is the rank of the document in the ordered dataset, which is the last manually screened document in order to achieve r% of recall.',
 'sec41_min_rank': 'nr% = N −(1 −r) · |I|. The maximum value of WSS is when the rank is equal to r% of relevant documents: nr% = r ·|I|.',
 # Run 2: the sentence of the Conclusions is split by the margin line number 290 in the extracted text,
 # so it is searched as two fragments (the text between them is only that line number).
 'conclusion_suggest_part1': 'We suggest the usage of TNR at',
 'conclusion_suggest_part2': 'r% of recall as an evaluation measure for the citation screening task',
}
Q['conclusion_suggest'] = Q['conclusion_suggest_part1'] + ' [margin line number 290] ' + Q['conclusion_suggest_part2']
loc = {k: locate(q) for k, q in Q.items() if k != "conclusion_suggest"}

# Exact checks
a_ok = all(n - (n * Fraction(1, 20)).__floor__() == math.ceil(Fraction(19, 20) * n) for n in range(1, 100001))
b_set = [n for n in range(1, 1001) if math.ceil(Fraction(19, 20) * n) == n]
b_ok = b_set == list(range(1, 20))
c_nonint = sum(1 for n in range(1, 1001) if (n * Fraction(1, 20)).denominator != 1)

ms = open(MS).read()
ms_k95 = 'WSS@95=1−k95/N−0.05, where k95 is the rank of the ⌈0.95n⌉-th of the n included records, so that in reviews with at most 19 included records 95% recall means all included records'
ms_tnr = 'The true negative rate at 95% recall (TNR@95) is the proportion of excluded records not yet read when the ⌈0.95n⌉-th included record is found'
ms_caveat = 'only the abstract of that article could be read, so its rounding of the 95% recall point was not compared with this definition (post hoc, analysis 50)'
for s in (ms_k95, ms_tnr, ms_caveat):
    assert s in ms, s
tn = open(TN).read()
assert 'only the abstract of that article could be' in tn

cr = json.load(open(CR))['message']
first_page = npages[1]
bib = {
 'crossref_title': cr['title'][0], 'crossref_journal': cr['container-title'][0], 'crossref_volume': cr.get('volume'),
 'crossref_article_number': cr.get('article-number'), 'crossref_issued': cr['issued']['date-parts'][0],
 'crossref_authors': [f"{a['given']} {a['family']}" for a in cr['author']], 'crossref_doi': cr['DOI'],
 'pdf_title_matches_crossref_casefold': norm(cr['title'][0]).casefold() in first_page.casefold(),
 'pdf_doi_on_cover': 'https://doi.org/10.1016/j.iswa.2023.200193' in first_page,
 'pdf_authors_on_cover': 'Wojciech Kusa, Aldo Lipani, Petr Knoth and Allan Hanbury' in first_page,
 'pdf_reference_ISWA_200193': 'ISWA 200193' in first_page,
 'pdf_dates_on_cover': {'received': '30 May 2022', 'revised': '30 November 2022', 'accepted': '29 January 2023'},
 'pdf_dates_found': all(x in first_page for x in ('30 May 2022', '30 November 2022', '29 January 2023')),
 'note': 'The pre-proof carries no volume number; volume 18 is from the Crossref record. The version of record was not obtained, so differences between the pre-proof and the version of record cannot be excluded.',
}

comparisons = [
 {'id': 'D1', 'topic': 'Rounding of the 95% recall point',
  'article_location': 'Section 3.3 (The FN term), printed page 7 (PDF page 9); notation table of Section 1.1 (nr%); Section 4.1 (nr%)',
  'article_quotes': [Q['sec33_floor'], Q['notation_nr'], Q['sec41_nr']],
  'manuscript_text': ms_k95,
  'judgement': 'agree',
  'reasoning': 'With FN = floor(0.05|I|), the number of included records found at the 95% recall point is |I| - floor(0.05|I|), which equals ceil(0.95|I|) for every integer |I| (exact check (a), n = 1..100000). The recall point nr% is the rank of the last record read to reach this recall (Section 4.1), that is the rank of the ceil(0.95n)-th included record, which is k95 of the Methods.',
  'exact_check': 'a'},
 {'id': 'D2', 'topic': 'nWSS@95% = TN@95%/|E| = TNR@95%',
  'article_location': 'Section 4 (The Normalised WSS), which begins on printed page 9 (PDF page 11); Equations 7 to 11 on printed page 10 (PDF page 12); Highlights (PDF page 2); Section 1 (printed page 4, PDF page 6); Conclusions (printed page 18, PDF page 20)',
  'article_quotes': [Q['sec4_eq9_text'], Q['sec4_eq10_text'], Q['sec4_eq11_text'], Q['highlight_tnr'], Q['conclusion_suggest']],
  'article_equations_transcribed': 'Eq. 8: nWSS@r% = TN/|E| (the floor terms cancel); Eq. 9: nWSS@95% = TN@95%/|E|; Eq. 10: nWSS = TN/(TN+FP); Eq. 11: nWSS@r% = TNR@r% = TN@r%/|E|. Transcribed by the AI tool from the extracted text, in which the fraction bars are lost.',
  'manuscript_text': ms_tnr,
  'judgement': 'agree',
  'reasoning': 'TN@95% counts the excluded records ranked after the recall point, that is not yet read when the ceil(0.95n)-th included record is found; dividing by |E| gives the proportion of excluded records not yet read, which is the definition of the Methods. The manuscript sentence "Normalised WSS has been shown to be equivalent to the true negative rate, and TNR at 95% recall has been proposed as a measure" is supported by Equations 8 to 11, the Highlights, Section 1 and the Conclusions.'},
 {'id': 'D3', 'topic': 'Reviews with fewer than 20 included records',
  'article_location': 'Section 3.5 (Evaluation with Cross-Validation), which begins on printed page 8 (PDF page 10); the quoted sentences and Equation 6 on printed page 9 (PDF page 11)',
  'article_quotes': [Q['sec35_zero_fn'], Q['sec35_eq6'], Q['sec35_lt20']],
  'manuscript_text': 'in reviews with at most 19 included records 95% recall means all included records (Methods, Outcomes); Because ⌈0.95n⌉=n for every n of at most 19 (Supplementary Table 23 note)',
  'judgement': 'agree',
  'reasoning': 'Fewer than 20 is at most 19. With no false negative allowed, the 95% recall point is the last included record, so WSS@95 = WSS@100 - 0.05 under the definitions of the Methods (WSS@95 = 1 - k95/N - 0.05 and WSS@100 = 1 - k100/N with k95 = k100), which is Equation 6. Exact check (b): ceil(0.95n) = n exactly for n = 1..19 and for no n from 20 to 1000.',
  'exact_check': 'b'},
 {'id': 'D4', 'topic': 'Floor-less expressions of Section 4.1 (point raised by the referee)',
  'article_location': 'Section 4.1, Equations 12 to 16, printed pages 11 and 12 (PDF pages 13 and 14); also the last sentence of Section 3.3',
  'article_quotes': [Q['sec41_min_rank'], Q['sec33_tp']],
  'manuscript_text': ms_k95,
  'judgement': 'differ (within the article; not a difference from the Methods definition)',
  'reasoning': 'Section 4.1 writes the extreme ranks as N - (1-r)|I| and r|I|, and Section 3.3 writes TP = r|I|, without the floor of Section 3.3; these are integers only when 0.05|I| is an integer (not the case for %d of n = 1..1000, exact check (c)). The rank-based result (Eq. 16) reduces to TN/(TN+FP) under the assumption TP = r|I|, the same expression as Eq. 10. The Methods define the recall point with the ceiling, which is consistent with the floor of Section 3.3 (D1) and not with the floor-less expressions. This does not change any value of the manuscript, whose TNR@95 is computed directly as a proportion of excluded records.' % c_nonint,
  'exact_check': 'c'},
]

res = {
 'analysis_id': 'AN-0003-01', 'manuscript_analysis_number': 51, 'post_hoc': True,
 'comment_ids': ['E3.09', 'R1.3.03', 'H3.07'],
 'full_text_obtained': True,
 'source': {'url': 'https://discovery.ucl.ac.uk/id/eprint/10164959/1/1-s2.0-S2667305323000182-main.pdf',
            'http_status': 200, 'fetch_record': 'fetch_record.tsv',
            'file': 'sources/ucl_eprint_10164959_kusa_main.pdf', 'extract_meta': 'extract_meta.json'},
 'record_page_request': {'url': 'https://discovery.ucl.ac.uk/id/eprint/10164959/', 'http_status': 403,
                         'note': 'Access-control page; not bypassed. The record page was not needed once the PDF was obtained.'},
 'version': {'kind': 'Journal pre-proof (Elsevier cover page), not the version of record',
             'quotes': [Q['cover_version'], Q['cover_not_vor']], 'pages_pdf': 26,
             'printed_pages': 'article text on printed pages 1 to 22 after a cover page and a Highlights page; declaration of interests and author contributions on the last two PDF pages'},
 'bibliographic_check': bib,
 'quotation_pages_pdf': loc,
 'exact_checks': {
   'a_n_minus_floor_equals_ceil_n_1_to_100000': a_ok,
   'b_ceil_095n_equals_n_exactly_for_n_le_19_within_1_to_1000': b_ok,
   'b_values': b_set,
   'c_n_1_to_1000_with_non_integer_005n': c_nonint},
 'comparisons': comparisons,
 'summary': 'The journal pre-proof was obtained from UCL Discovery (HTTP 200). Its definitions of the 95% recall point (Section 3.3, floor of the number of false negatives), of nWSS@95% = TN@95%/|E| = TNR@95% (Equations 9 to 11), and of the equivalence of 95% and 100% recall for fewer than 20 included records (Section 3.5) agree with the definitions of the Methods (Outcomes). Section 4.1 and the last sentence of Section 3.3 use floor-less expressions (TP = r|I|) that are exact only when 0.05|I| is an integer; the Methods follow the floor of Section 3.3. The version of record was not obtained. The comparison was made by the AI coding tool and has not been verified by an author.',
 'manuscript_texts_checked': {'methods_outcomes_k95': ms_k95, 'methods_outcomes_tnr': ms_tnr, 'methods_outcomes_caveat_v0002': ms_caveat,
                              'source_manuscript': MS, 'source_si_table_23_note': TN},
 'network': 'none in this script',
}
json.dump(res, open(f'{D}/results.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps({k: res[k] for k in ('full_text_obtained', 'exact_checks', 'summary')}, indent=1, ensure_ascii=False))
print({c['id']: c['judgement'] for c in comparisons})
print({k: v for k, v in bib.items() if k.startswith('pdf_')})
