#!/usr/bin/env python3
"""AN-0002-04 (manuscript analysis 50, post hoc): verification of three sources against saved public text.
Reads only files saved by fetch.sh in sources/ (this folder), the saved arXiv text of round 1
(AN-0001-14/sources/arxiv_pdf_2212.09017_wang.txt), the v0001 reference list, and
AN-0001-01/results.json (records per percentage point). Writes results.json. No network access."""
import json, re, pathlib
import lxml.etree as E
import lxml.html as H

D = pathlib.Path('/N/project/AiLab/jev/review_pipeline_npj/rounds/round_0002/revision/analysis/AN-0002-04')
S = D / 'sources'
R1 = pathlib.Path('/N/project/AiLab/jev/review_pipeline_npj/rounds/round_0001/revision/analysis')
V1 = pathlib.Path('/N/project/AiLab/jev/review_pipeline_npj/versions/v0001_after_round0001/source')
ws = lambda s: re.sub(r'\s+', ' ', s).strip()

def find(txt, needle, before=0, after=0):
    i = txt.find(needle)
    if i < 0:
        return None
    return txt[max(0, i - before): i + len(needle) + after]

def crossref(fn):
    m = json.load(open(S / fn))['message']
    return {'authors': [f"{a.get('family')}, {a.get('given')}" for a in m.get('author', [])],
            'title': m['title'][0], 'journal': m.get('container-title', [None])[0], 'volume': m.get('volume'),
            'article_number': m.get('article-number'), 'page': m.get('page'),
            'issued': m['issued']['date-parts'][0], 'doi': m['DOI'],
            'licence_vor': [l['URL'] for l in m.get('license', []) if l.get('content-version') == 'vor']}

out = {'analysis': 'AN-0002-04', 'manuscript_analysis_number': 50, 'post_hoc': True}

# ---------------------------------------------------------------- (1) Kusa, Lipani, Knoth & Hanbury 2023
k = {'crossref': crossref('crossref_10.1016_j.iswa.2023.200193.json')}
k['crossref_record_identical_to_round1_copy'] = (S / 'crossref_10.1016_j.iswa.2023.200193.json').read_bytes() == \
    (R1 / 'AN-0001-14/sources/crossref_10.1016_j.iswa.2023.200193.json').read_bytes()
rep = ws(H.parse(str(S / 'repositum_188936_full.html')).getroot().text_content())
abs_txt = find(rep, 'Citation screening is an essential', 0, 0)
i = rep.find('Citation screening is an essential'); j = rep.find('compare the measure with Precision and AUC.')
abstract = rep[i: j + len('compare the measure with Precision and AUC.')]
k['abstract_source'] = 'TU Wien reposiTUm record 20.500.12708/188936 (?mode=full), field dc.description.abstract; sources/repositum_188936_full.html'
k['abstract'] = abstract
k['quote_equivalence'] = find(abstract, 'We analytically show that normalised WSS is equivalent to the True Negative Rate (TNR).')
k['quote_tnr95'] = find(abstract, 'we provide benchmark scores for fifteen systematic review datasets with TNR@95% recall measure')
k['quote_normalise'] = find(abstract, 'We subsequently propose to normalise WSS which enables citation screening performance comparisons across different systematic reviews.')
k['full_text_retrieved'] = False
k['full_text_attempts'] = {
    'UCL Discovery landing page': 'HTTP 403 (Cloudflare check page), sources/ucl_discovery_10164959.html',
    'UCL Discovery PDF (listed by OpenAlex)': 'HTTP 404, sources/ucl_10164959_kusa.pdf (HTML error page)',
    'Open Research Online PDF (listed by OpenAlex)': 'HTTP 403, sources/oro_87644_kusa.pdf (HTML error page)',
    'TU Wien reposiTUm': 'HTTP 200, bibliographic record and abstract only; no file attached',
    'ScienceDirect article page': 'HTTP 403, sources/sciencedirect_S2667305323000182.html',
    'Elsevier article API (TDM link given by Crossref), text/xml without key': 'HTTP 200, bibliographic data only (no body), sources/elsevier_api_S2667305323000182_kusa.xml',
    'Semantic Scholar': 'openAccessPdf points to the DOI landing page (GOLD, CC BY)'}
els = (S / 'elsevier_api_S2667305323000182_kusa.xml').read_text()
k['publisher_open_access_flag'] = {'openaccessArticle': re.search(r'<openaccessArticle>(.*?)<', els).group(1),
                                   'licence': re.search(r'<openaccessUserLicense>(.*?)<', els).group(1)}
k['verified'] = {
    'normalised WSS is equivalent to the true negative rate (the authors show it analytically)': 'YES (abstract)',
    'the paper reports TNR at 95% recall as a measure': 'YES (abstract)',
    'definition of normalised WSS or TNR@95% recall in detail, including how the 95% recall point is rounded (k95)': 'NOT VERIFIED (full text not retrieved)'}
refs = json.load(open(V1 / 'references_npjdm.json'))
k['existing_reference_kusa_in_v0001'] = refs.get('kusa')
k['existing_kusa_is_a_different_paper'] = ('Outcome-based evaluation' in refs.get('kusa', '')) and \
    ('work saved over sampling' in k['crossref']['title'].lower())
k['nature_style_reference_from_crossref'] = ('Kusa, W., Lipani, A., Knoth, P. & Hanbury, A. An analysis of work saved over sampling '
                                           'in the evaluation of automated citation screening in systematic literature reviews. '
                                           f"*Intell. Syst. Appl.* **{k['crossref']['volume']}**, {k['crossref']['article_number']} ({k['crossref']['issued'][0]}).")
out['kusa_lipani_2023'] = k

# ---------------------------------------------------------------- (2) Wallace et al. 2010
w = {'crossref': crossref('crossref_10.1186_1471-2105-11-55.json')}
pm = E.parse(str(S / 'pubmed_efetch_20102628.xml'))
art = pm.find('.//PubmedArticle')
w['pubmed'] = {'pmid': art.findtext('.//PMID'), 'title': art.findtext('.//ArticleTitle'),
               'journal': art.findtext('.//Journal/Title'), 'volume': art.findtext('.//JournalIssue/Volume'),
               'pages': art.findtext('.//Pagination/MedlinePgn'), 'year': art.findtext('.//PubDate/Year'),
               'authors': [f"{a.findtext('LastName')} {a.findtext('Initials')}" for a in art.findall('.//AuthorList/Author')],
               'pmcid': [x.text for x in art.findall('.//PubmedData/ArticleIdList/ArticleId') if x.get('IdType') == 'pmc']}
pmc = E.parse(str(S / 'pmc_efetch_PMC2824679_wallace.xml'))
target = 'An experienced reviewer can screen an average of two abstracts per minute.'
loc = None
for p in pmc.iter('p'):
    t = ws(' '.join(p.itertext()))
    if target in t:
        sec = p.getparent(); top = sec.getparent()
        ps = [x for x in sec if x.tag == 'p']
        loc = {'section': ws(top.findtext('title') or ''), 'subsection': ws(sec.findtext('title') or ''),
               'paragraph': f'{ps.index(p) + 1} of {len(ps)}', 'paragraph_text': t,
               'citations_in_paragraph': [x.get('rid') for x in p.iter('xref')]}
w['full_text_source'] = 'PMC2824679 via PubMed Central efetch; sources/pmc_efetch_PMC2824679_wallace.xml'
w['quote_rate'] = target if loc else None
w['location'] = loc
w['rate_is_authors_statement_without_citation'] = bool(loc) and not loc['citations_in_paragraph']
w['context_caveat'] = 'Abstracts for difficult topics may take several minutes each to evaluate' if loc and \
    'Abstracts for difficult topics may take several minutes each to evaluate' in loc['paragraph_text'] else None
w['verified'] = {'an experienced reviewer screens an average of two abstracts per minute': 'YES (full text, Background)' if loc else 'NO',
                 'the rate is cited from another source': 'NO: no citation in the paragraph; it is stated by the authors without a source' if w['rate_is_authors_statement_without_citation'] else 'see location'}
w['nature_style_reference_from_crossref'] = ('Wallace, B. C., Trikalinos, T. A., Lau, J., Brodley, C. & Schmid, C. H. Semi-automated screening '
                                           f"of biomedical citations for systematic reviews. *BMC Bioinformatics* **{w['crossref']['volume']}**, "
                                           f"{w['crossref']['article_number']} ({w['crossref']['issued'][0]}).")
out['wallace_2010'] = w

# ---------------------------------------------------------------- illustrative conversion (E2.14)
a01 = json.load(open(R1 / 'AN-0001-01/results.json'))
pp = a01['c_cost']['records_relations']['heldout']['one_percentage_point_of_records']
rate = 2.0  # abstracts per minute, Wallace et al. 2010 (verified above)
out['illustrative_conversion'] = {
    'source_of_records': 'AN-0001-01/results.json c_cost.records_relations.heldout.one_percentage_point_of_records (analysis 32)',
    'records_per_percentage_point_heldout': pp,
    'rate_abstracts_per_minute': rate,
    'minutes_per_percentage_point': pp / rate,
    'hours_per_percentage_point': pp / rate / 60,
    'label': 'illustrative; the rate comes from another setting (Wallace et al. 2010) and was not measured in this study'} if loc else None

# ---------------------------------------------------------------- (3) Wang et al. arXiv:2212.09017 (ref. wang_neural)
wt = (R1 / 'AN-0001-14/sources/arxiv_pdf_2212.09017_wang.txt').read_text()
wtn = ws(wt.replace('-\n', ''))
q = {}
q['source'] = 'round 1 saved text of the arXiv PDF: rounds/round_0001/revision/analysis/AN-0001-14/sources/arxiv_pdf_2212.09017_wang.txt (no re-fetch)'
q['quote_query'] = find(wtn, 'We use the title of the review for each topic as the query to rank documents.')
q['quote_training'] = find(wtn, 'we further fine-tune the BERT model using the training portion in our dataset and then apply the resulting ranker on the screening prioritisation task on the test portion of the dataset.')
q['quote_labels_for_training'] = find(wtn, 'is a document judged relevant at the abstract level')
q['quote_datasets'] = find(wtn, 'We use three CLEF Technological Assisted Review (TAR) datasets')
q['quote_splits'] = find(wtn, 'The CLEF TAR datasets contain 50 systematic review topics in 2017 (20 training, 30 testing)', 0, 300)
q['quote_zero_shot'] = find(wtn, 'Overall, the use of zero-shot neural rankers for the task of screening prioritisation does not appear to be a competitive and viable approach to the task.')
q['quote_fine_tuned'] = find(wtn, 'rankers fine-tuned on even a small amount of training data achieve significantly higher effectiveness than the current state-of-the-art non-iterative methods')
q['quote_truncation'] = find(wtn, 'BERT has an input limit of 512 tokens', 0, 200)
q['section_query_and_datasets'] = '4.1 Dataset & Evaluation'
q['section_training'] = '3.1 Model Architecture'
q['verified'] = {'query was the title of the review': 'YES' if q['quote_query'] else 'NO',
                 'fine-tuned rankers were trained on the training portions (topics) of the CLEF TAR 2017, 2018 and 2019 collections, with abstract-level relevance labels': 'YES' if (q['quote_training'] and q['quote_labels_for_training'] and q['quote_splits']) else 'NO',
                 'zero-shot neural rankers were not competitive': 'YES' if q['quote_zero_shot'] else 'NO',
                 'fine-tuned rankers were effective': 'YES' if q['quote_fine_tuned'] else 'NO'}
q['wording_supported'] = "'neural rankers fine-tuned on screening labels of other reviews (the training topics of the CLEF TAR collections) were effective'"
out['wang_2212_09017'] = q

json.dump(out, open(D / 'results.json', 'w'), indent=1, ensure_ascii=False)
for key in ['kusa_lipani_2023', 'wallace_2010', 'wang_2212_09017']:
    print(key, json.dumps(out[key]['verified'], ensure_ascii=False))
print('conversion', json.dumps(out['illustrative_conversion']))
