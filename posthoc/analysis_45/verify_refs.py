# AN-0001-14: build the reference verification table from the responses saved in sources/ (no network access needed to re-run).
# Every supporting sentence is checked for presence in the saved source text (exact match after whitespace and hyphen normalisation,
# and a secondary match on lower-case letters and digits only, used for PDF text with line-break hyphenation and for Unicode minus signs).
# Run: PYTHONNOUSERSITE=1 /N/project/AiLab/jev/synergy/.venv/bin/python verify_refs.py
import json, os, re, csv, html, xml.etree.ElementTree as ET

H = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(H, 'sources')


def rd(name):
    return open(os.path.join(S, name), encoding='utf8').read()


def norm(t):
    t = html.unescape(t)
    for a, b in (('‐', '-'), ('‑', '-'), ('−', '-'), ('’', "'"), (' ', ' ')):
        t = t.replace(a, b)
    return re.sub(r'\s+', ' ', t).strip()


def alnum(t):
    return re.sub(r'[^a-z0-9]', '', html.unescape(t).lower())


def strip_tags(t):
    return re.sub(r'<[^>]+>', ' ', t)


def crossref(doi):
    m = json.load(open(os.path.join(S, 'crossref_' + doi.replace('/', '_') + '.json')))['message']
    dp = lambda k: m.get(k, {}).get('date-parts', [[None]])[0]
    return {'type': m.get('type'), 'title': (m.get('title') or [''])[0], 'container': (m.get('container-title') or [''])[0],
            'volume': m.get('volume'), 'issue': m.get('issue'), 'page': m.get('page'), 'article_number': m.get('article-number'),
            'published': dp('published'), 'published_print': dp('published-print'), 'published_online': dp('published-online'),
            'authors': [((a.get('given') or '') + ' ' + (a.get('family') or '')).strip() for a in m.get('author', [])],
            'publisher': m.get('publisher'), 'abstract': norm(strip_tags(m['abstract'])) if m.get('abstract') else None}


pubmed = {}
for f in ('pubmed_efetch_refs.xml', 'pubmed_efetch_19381330_higgins.xml', 'pubmed_efetch_screening_time.xml'):
    for a in ET.parse(os.path.join(S, f)).getroot().findall('.//PubmedArticle'):
        j = a.find('.//Journal')
        pubmed[a.findtext('.//PMID')] = {
            'file': f, 'title': a.findtext('.//ArticleTitle'), 'journal': j.findtext('.//ISOAbbreviation'),
            'volume': j.findtext('.//JournalIssue/Volume'), 'issue': j.findtext('.//JournalIssue/Issue'),
            'year': j.findtext('.//JournalIssue/PubDate/Year'), 'pages': a.findtext('.//MedlinePgn'),
            'elocation': [e.text for e in a.findall('.//ELocationID')],
            'authors': [(x.findtext('LastName') or '') + ' ' + (x.findtext('Initials') or '') for x in a.findall('.//AuthorList/Author')],
            'ids': {e.get('IdType'): e.text for e in a.findall('.//PubmedData/ArticleIdList/ArticleId')},
            'abstract': norm(' '.join(''.join(t.itertext()) for t in a.findall('.//Abstract/AbstractText')))}

texts = {
    'pubmed:42767575': pubmed['42767575']['abstract'], 'pubmed:41230118': pubmed['41230118']['abstract'],
    'pubmed:31972274': pubmed['31972274']['abstract'], 'pubmed:38429798': pubmed['38429798']['abstract'],
    'pubmed:37930897': pubmed['37930897']['abstract'], 'pubmed:19381330': pubmed['19381330']['abstract'],
    'europepmc_fulltext:PMC12603384': norm(strip_tags(rd('europepmc_fulltext_PMC12603384.xml'))),
    'pmc_fulltext:PMC2667312': norm(strip_tags(rd('pmc_efetch_PMC2667312_higgins.xml'))),
    'crossref_abstract:10.1145/3631990': crossref('10.1145/3631990')['abstract'],
    'arxiv_pdf_text:2212.09017v1': norm(rd('arxiv_pdf_2212.09017_wang.txt')),
    'hf_model_card:ncbi/MedCPT-Cross-Encoder': norm(rd('hf_ncbi_MedCPT-Cross-Encoder_README.md')),
}
for f, k in (('acl_2023.findings-emnlp.171.html', 'acl_abstract:2023.findings-emnlp.171'), ('acl_2023.emnlp-main.330.html', 'acl_abstract:2023.emnlp-main.330')):
    m = re.search(r'Abstract</h5><span>(.*?)</span>', rd(f), re.S)
    texts[k] = norm(strip_tags(m.group(1))) if m else ''
m = re.search(r'<meta name="citation_abstract" content="([^"]*)"', rd('arxiv_abs_2212.09017_wang.html'))
texts['arxiv_abstract:2212.09017'] = norm(m.group(1))


def check(src, quote):
    t = texts[src]
    exact = norm(quote) in t
    loose = alnum(quote) in alnum(t)
    return {'source': src, 'quote': quote, 'exact_match': exact, 'alnum_match': loose}


# Lancet copy of the ref. 14 abstract (read-only) for a consistency check
lancet = ET.parse('/N/project/AiLab/jev/review_pipeline/rounds/round_0001/revision/analysis/AN-0001-18/api_responses/pubmed_efetch_all.xml').getroot()
lancet_abs = None
for a in lancet.findall('.//PubmedArticle'):
    if a.findtext('.//PMID') == '42767575':
        lancet_abs = norm(' '.join(''.join(t.itertext()) for t in a.findall('.//Abstract/AbstractText')))

refs = []
# ---- ref. 14 (v0 'pitre'): numbers quoted in the Introduction of v0 ----
refs.append({
    'key': 'pitre (v0 ref. 14)', 'doi': '10.1016/j.jclinepi.2026.112514', 'pmid': '42767575', 'in_v0': True,
    'crossref': crossref('10.1016/j.jclinepi.2026.112514'), 'pubmed': {k: v for k, v in pubmed['42767575'].items() if k != 'abstract'},
    'full_text_reached': False,
    'full_text_attempts': 'doi.org -> linkinghub (redirect page only), ScienceDirect article page 403, Elsevier article API 400 without key, Europe PMC: no free full text (isOpenAccess N, inEPMC N), SSRN preprint page 403',
    'support': [
        check('pubmed:42767575', 'We assembled 762,934 citation-label pairs from 19,787 completed reviews and trained TITAN-SR on a prespecified review-level split, testing on held-out reviews.'),
        check('pubmed:42767575', 'developed from nearly 20,000 completed reviews'),
        check('pubmed:42767575', 'Across 3,434 test reviews, TITAN-SR achieved a median specificity at 99% recall of 0.902 and median area under the receiver operating characteristic curve (AUC) of 0.974.'),
        check('pubmed:42767575', 'On the 22 external reviews performance was virtually identical (median AUC 0.976; specificity at 95% recall 0.897), and at a threshold fixed on the internal validation split TITAN-SR met the pre-registered primary criterion, retaining 99.8% of included studies (95% confidence interval [CI] 99.5 to 100.0%).'),
        check('pubmed:42767575', 'with ASReview on 22 temporally independent Cochrane reviews (142,504 records); ASReview received a warm-up of 20% of known includes.'),
        check('pubmed:42767575', 'Temporal validation on 22 held-out Cochrane reviews confirmed generalization to reviews published after the model-development period.'),
    ],
    'v0_claims': [
        {'v0_text': 'required training on almost 20,000 completed reviews', 'source_value': '19,787 completed reviews; "developed from nearly 20,000 completed reviews"', 'verdict': 'SUPPORTED'},
        {'v0_text': 'it reported a median AUC of 0.974 on its own test set', 'source_value': 'median AUC 0.974 across 3,434 test reviews (internal held-out test reviews)', 'verdict': 'SUPPORTED'},
        {'v0_text': 'with a threshold fixed on its validation split, retained 99.8% of included studies in 22 external reviews', 'source_value': 'The abstract gives 99.8% (95% CI 99.5 to 100.0%) at a threshold fixed on the internal validation split, in the sentence that begins "On the 22 external reviews"; the full text could not be reached to confirm the set of reviews and the pooling (for example, whether 99.8% is pooled over studies or a median over reviews).', 'verdict': 'SUPPORTED BY THE ABSTRACT (full text not reached)'},
        {'v0_text': 'outperformed active learning and chatbot LLMs', 'source_value': 'title: "outperformed active learning and LLM chatbots"; ASReview received a warm-up of 20% of known includes', 'verdict': 'SUPPORTED (comparison conditions differ from this study)'},
    ],
    'lancet_copy_identical': lancet_abs == pubmed['42767575']['abstract'],
    'status': 'VERIFIED_ABSTRACT_ONLY',
    'nature_reference_draft': 'unchanged from v0 (references_npjdm.json key pitre); Crossref today still shows no volume, only article number 112514',
})

# ---- new references ----
refs.append({
    'key': 'flemyng', 'doi': '10.1002/cl2.70074', 'pmid': '41230118', 'pmcid': 'PMC12603384',
    'crossref': crossref('10.1002/cl2.70074'), 'pubmed': {k: v for k, v in pubmed['41230118'].items() if k != 'abstract'}, 'full_text_reached': True,
    'support': [
        check('pubmed:41230118', 'AI and automation in evidence synthesis should be used with human oversight.'),
        check('pubmed:41230118', 'Any use of AI or automation that makes or suggests judgements should be fully and transparently reported in the evidence synthesis report.'),
        check('europepmc_fulltext:PMC12603384', 'The name(s) of the AI system(s), tool(s) or platform(s), version(s), and date(s) used.'),
        check('europepmc_fulltext:PMC12603384', 'and how it has been validated (and piloted, if applicable) to ensure that it is appropriate for use in the context of the specific evidence synthesis.'),
        check('europepmc_fulltext:PMC12603384', 'they may need to pilot (or calibrate) the AI system or tool to validate its performance within their evidence synthesis, to ensure its use will not undermine the trustworthiness or reliability of the synthesis or its conclusions.'),
        check('pubmed:41230118', 'Cochrane, the Campbell Collaboration, JBI, and the Collaboration for Environmental Evidence support the aims of the Responsible use of AI in evidence SynthEsis (RAISE) recommendations'),
    ],
    'status': 'VERIFIED',
    'supports_use': 'human oversight; transparent reporting of AI use that makes or suggests judgements; reporting of tool name, version and dates; validation or piloting of a tool in the context of the specific synthesis.',
    'nature_reference_draft': 'Flemyng, E. et al. Position statement on artificial intelligence (AI) use in evidence synthesis across Cochrane, the Campbell Collaboration, JBI, and the Collaboration for Environmental Evidence 2025. *Campbell Syst. Rev.* **21**, e70074 (2025).',
})
refs.append({
    'key': 'kusa_wss', 'doi': '10.1016/j.iswa.2023.200193', 'pmid': None,
    'crossref': crossref('10.1016/j.iswa.2023.200193'), 'full_text_reached': False,
    'full_text_attempts': 'Crossref record has no abstract; doi.org -> linkinghub redirect only; ScienceDirect article page and PDF endpoint 403; Elsevier article API 400 without key; no arXiv version (arXiv title search); not in Europe PMC (0 hits); CORE search 403; TU Wien reposiTUm 416',
    'support': [],
    'status': 'EXISTS_BIBLIO_VERIFIED_CONTENT_NOT_VERIFIED',
    'supports_use': 'Existence, authors (Kusa, Lipani, Knoth, Hanbury), title, journal, volume 18, article 200193, May 2023, CC BY licence verified in Crossref. The statement that normalised WSS equals the true negative rate at the target recall was NOT verified from the source (no abstract or full text retrieved). The reviewers and editor give it; under I-8 it cannot be attributed to this paper in the text until the full text is read.',
    'nature_reference_draft': 'Kusa, W., Lipani, A., Knoth, P. & Hanbury, A. An analysis of work saved over sampling in the evaluation of automated citation screening in systematic literature reviews. *Intell. Syst. Appl.* **18**, 200193 (2023).',
})
refs.append({
    'key': 'wang_neural', 'doi': '10.1145/3572960.3572980', 'arxiv': '2212.09017v1',
    'crossref': crossref('10.1145/3572960.3572980'), 'full_text_reached': 'arXiv version 1 (18 December 2022; header ADCS \'22); ACM version not read',
    'support': [
        check('arxiv_abstract:2212.09017', 'In this paper, we apply several pre-trained language models to the systematic review document ranking task, both directly and fine-tuned.'),
        check('arxiv_pdf_text:2212.09017v1', 'In the zero-shot setting, we use the pre-trained language models directly on the screening prioritisation task.'),
        check('arxiv_pdf_text:2212.09017v1', 'We use the title of the review for each topic as the query to rank documents.'),
        check('arxiv_pdf_text:2212.09017v1', 'In the fine-tuned setting, we further fine-tune the BERT model using the training portion in our dataset'),
        check('arxiv_pdf_text:2212.09017v1', 'Overall, the use of zero-shot neural rankers for the task of screening prioritisation does not appear to be a competitive and viable approach to the task.'),
        check('arxiv_abstract:2212.09017', 'Our results show that BERT-based rankers outperform the current state-of-the-art screening prioritisation methods.'),
    ],
    'status': 'VERIFIED',
    'supports_use': 'Neural (monoBERT cross-encoder) rankers were applied to screening prioritisation zero-shot, with the review title as query, and after fine-tuning on the abstract-level labels of CLEF TAR training topics. The zero-shot rankers were not competitive with BM25 and QLM; the fine-tuned rankers outperformed prior methods. A gap statement must therefore say that zero-shot rankers of the review topic exist (this study and Wang et al.) and must not describe the effective rankers of Wang et al. as free of training on screening labels.',
    'nature_reference_draft': 'Wang, S., Scells, H., Koopman, B. & Zuccon, G. Neural rankers for effective screening prioritisation in medical systematic review literature search. In *Proc. 26th Australasian Document Computing Symposium* 1–10 (ACM, 2022).',
})
acl171 = rd('acl_2023.findings-emnlp.171.bib')
refs.append({
    'key': 'binhezam', 'doi': '10.18653/v1/2023.findings-emnlp.171',
    'crossref': crossref('10.18653/v1/2023.findings-emnlp.171'), 'acl_anthology_bib': norm(acl171), 'full_text_reached': False,
    'support': [
        check('acl_abstract:2023.findings-emnlp.171', 'This paper extends an effective stopping rule using information derived from a text classifier that can be trained without the need for any additional annotation.'),
        check('acl_abstract:2023.findings-emnlp.171', 'Experiments on multiple data sets (CLEF e-Health, TREC Total Recall, TREC Legal and RCV1) showed that the proposed approach consistently improves performance and outperforms several alternative methods.'),
    ],
    'status': 'VERIFIED',
    'note': 'Crossref gives the first author family name as "Hezam" and no pages; the ACL Anthology BibTeX gives "Bin-Hezam, Reem" and pages 2603--2609. The ACL Anthology form is used. The words "counting process" come from the title.',
    'supports_use': 'a stopping rule (counting process, per the title) extended with information from a text classifier trained without additional annotation.',
    'nature_reference_draft': 'Bin-Hezam, R. & Stevenson, M. Combining counting processes and classification improves a stopping rule for technology assisted review. In *Findings of the Association for Computational Linguistics: EMNLP 2023* 2603–2609 (Association for Computational Linguistics, 2023).',
})
refs.append({
    'key': 'stevenson', 'doi': '10.1145/3631990',
    'crossref': {k: v for k, v in crossref('10.1145/3631990').items() if k != 'abstract'}, 'full_text_reached': False,
    'support': [
        check('crossref_abstract:10.1145/3631990', 'This article proposes a novel stopping method based on point processes, which are statistical models that can be used to represent the occurrence of random events.'),
        check('crossref_abstract:10.1145/3631990', 'Results show that the proposed method achieves the desired level of recall without requiring an excessive number of documents to be examined in the majority of cases and also compares well against multiple alternative approaches.'),
    ],
    'status': 'VERIFIED',
    'supports_use': 'stopping methods for TAR based on point processes (rate functions of relevant documents in the ranking).',
    'nature_reference_draft': 'Stevenson, M. & Bin-Hezam, R. Stopping methods for technology-assisted reviews based on point processes. *ACM Trans. Inf. Syst.* **42**, 1–37 (2024).',
})
refs.append({
    'key': 'boetje', 'doi': '10.1186/s13643-024-02502-7', 'pmid': '38429798', 'pmcid': 'PMC10908130',
    'crossref': {k: v for k, v in crossref('10.1186/s13643-024-02502-7').items() if k != 'abstract'}, 'pubmed': {k: v for k, v in pubmed['38429798'].items() if k != 'abstract'}, 'full_text_reached': True,
    'support': [
        check('pubmed:38429798', 'This paper introduces the SAFE procedure, a practical and conservative set of stopping heuristics that offers a clear guideline for determining when to end the active learning process in screening software like ASReview.'),
    ],
    'status': 'VERIFIED',
    'supports_use': 'a practical, conservative set of stopping heuristics for active-learning screening in ASReview.',
    'nature_reference_draft': 'Boetje, J. & van de Schoot, R. The SAFE procedure: a practical stopping heuristic for active learning-based screening in systematic reviews and meta-analyses. *Syst. Rev.* **13**, 81 (2024).',
})
refs.append({
    'key': 'gartlehner', 'doi': '10.1016/j.jclinepi.2020.01.005', 'pmid': '31972274',
    'crossref': crossref('10.1016/j.jclinepi.2020.01.005'), 'pubmed': {k: v for k, v in pubmed['31972274'].items() if k != 'abstract'}, 'full_text_reached': False,
    'full_text_attempts': 'Europe PMC: no free full text (isOpenAccess N); publisher page not tried after the ScienceDirect 403 responses for the other Elsevier articles',
    'support': [
        check('pubmed:31972274', 'Overall, single-reviewer abstract screening missed 13% of relevant studies (sensitivity: 86.6%; 95% confidence interval [CI], 80.6%-91.2%).'),
        check('pubmed:31972274', 'By comparison, dual-reviewer abstract screening missed 3% of relevant studies (sensitivity: 97.5%; 95% CI, 95.1%-98.8%).'),
        check('pubmed:31972274', 'We calculated sensitivities and specificities of single- and dual-reviewer screening using two published systematic reviews as reference standards.'),
        check('pubmed:31972274', 'Two hundred and eighty participants made 24,942 screening decisions on 2,000 randomly selected abstracts from the reference standard reviews.'),
    ],
    'status': 'VERIFIED_ABSTRACT_ONLY',
    'note': 'The abstract does not state whether "relevant" in the reference standard means included at title and abstract level or finally included; the plan wording "concerns title and abstract relevance, not final inclusion" is therefore not verified. Safe wording: sensitivity of single-reviewer (86.6%) and dual-reviewer (97.5%) abstract screening by crowd participants against two published reviews as reference standards, with the note that the reference standard and the setting differ from the final-inclusion labels used here.',
    'supports_use': 'single-reviewer sensitivity 86.6% (80.6 to 91.2%), dual 97.5% (95.1 to 98.8%) in a crowd-based randomised trial.',
    'nature_reference_draft': 'Gartlehner, G. et al. Single-reviewer abstract screening missed 13 percent of relevant studies: a crowd-based, randomized controlled trial. *J. Clin. Epidemiol.* **121**, 20–28 (2020).',
})
acl330 = rd('acl_2023.emnlp-main.330.bib')
refs.append({
    'key': 'tian', 'doi': '10.18653/v1/2023.emnlp-main.330',
    'crossref': crossref('10.18653/v1/2023.emnlp-main.330'), 'acl_anthology_bib': norm(acl330), 'full_text_reached': False,
    'support': [
        check('acl_abstract:2023.emnlp-main.330', "For RLHF-LMs such as ChatGPT, GPT-4, and Claude, we find that verbalized confidences emitted as output tokens are typically better-calibrated than the model's conditional probabilities on the TriviaQA, SciQ, and TruthfulQA benchmarks, often reducing the expected calibration error by a relative 50%."),
    ],
    'status': 'VERIFIED',
    'supports_use': 'for RLHF language models, verbalised confidences were typically better calibrated than conditional (token) probabilities on three question-answering benchmarks.',
    'nature_reference_draft': 'Tian, K. et al. Just ask for calibration: strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In *Proc. 2023 Conference on Empirical Methods in Natural Language Processing* 5433–5442 (Association for Computational Linguistics, 2023).',
})
hf = json.load(open(os.path.join(S, 'hf_api_ncbi_MedCPT-Cross-Encoder.json')))
refs.append({
    'key': 'medcpt', 'doi': '10.1093/bioinformatics/btad651', 'pmid': '37930897', 'pmcid': 'PMC10627406',
    'crossref': crossref('10.1093/bioinformatics/btad651'), 'pubmed': {k: v for k, v in pubmed['37930897'].items() if k != 'abstract'}, 'full_text_reached': False,
    'full_text_attempts': 'Europe PMC fullTextXML 500; NCBI PMC efetch returned the front matter only (11,173 bytes)',
    'hf_model': {'id': 'ncbi/MedCPT-Cross-Encoder', 'sha': hf.get('sha'), 'lastModified': hf.get('lastModified'), 'license': (hf.get('cardData') or {}).get('license')},
    'support': [
        check('pubmed:37930897', 'To train MedCPT, we collected an unprecedented scale of 255 million user click logs from PubMed.'),
        check('pubmed:37930897', 'With such data, we use contrastive learning to train a pair of closely integrated retriever and re-ranker.'),
        check('hf_model_card:ncbi/MedCPT-Cross-Encoder', 'max_length=512'),
        check('hf_model_card:ncbi/MedCPT-Cross-Encoder', 'Higher scores indicate higher relevance.'),
    ],
    'status': 'VERIFIED',
    'supports_use': 'MedCPT cross-encoder (re-ranker) trained with contrastive learning on PubMed search click logs; model card usage truncates query-article pairs at 512 tokens; the model card asks users to cite the Bioinformatics paper. The model is trained on search logs, not on screening decisions.',
    'nature_reference_draft': 'Jin, Q. et al. MedCPT: contrastive pre-trained transformers with large-scale PubMed search logs for zero-shot biomedical information retrieval. *Bioinformatics* **39**, btad651 (2023).',
})
refs.append({
    'key': 'higgins', 'doi': '10.1111/j.1467-985X.2008.00552.x', 'pmid': '19381330', 'pmcid': 'PMC2667312',
    'crossref': crossref('10.1111/j.1467-985X.2008.00552.x'), 'pubmed': {k: v for k, v in pubmed['19381330'].items() if k != 'abstract'}, 'full_text_reached': True,
    'support': [
        check('pubmed:19381330', 'We propose a simple prediction interval for classical meta-analysis'),
        check('pmc_fulltext:PMC2667312', 'Again taking a t -distribution with k -2 degrees of freedom, we assume that, approximately,'),
        check('pmc_fulltext:PMC2667312', 'Thus an approximate 100(1- α )% prediction interval for the effect in an unspecified study can be obtained as'),
        check('pmc_fulltext:PMC2667312', 'percentile of the t -distribution with k -2 degrees of freedom'),
    ],
    'formula_images': {'expression_11': 'sources/pmc_PMC2667312_rssa0172-0137-m11.jpg: (theta_new - mu_hat) / sqrt(tau_hat^2 + SE(mu_hat)^2) ~ t_{k-2}',
                       'expression_12': 'sources/pmc_PMC2667312_rssa0172-0137-m12.jpg: mu_hat +/- t^alpha_{k-2} sqrt(tau_hat^2 + SE(mu_hat)^2)',
                       'how_read': 'The two expressions are images in the PMC record; they were read visually from the saved images by the analysing agent (Claude Opus 5.5). The surrounding text (t distribution with k-2 degrees of freedom) matches by string search.'},
    'status': 'VERIFIED',
    'supports_use': 'approximate 95% prediction interval for the effect in a new study: mu_hat +/- t_{k-2, 0.975} sqrt(tau_hat^2 + SE(mu_hat)^2), as planned for AN-0001-12.',
    'nature_reference_draft': 'Higgins, J. P. T., Thompson, S. G. & Spiegelhalter, D. J. A re-evaluation of random-effects meta-analysis. *J. R. Stat. Soc. Ser. A Stat. Soc.* **172**, 137–159 (2009).',
})

# ---- R4.1.14: pre-specified PubMed search for a per-record human screening time ----
es = json.load(open(os.path.join(S, 'pubmed_esearch_screening_time.json')))['esearchresult']
qstr = open(os.path.join(H, 'screening_time_search_string.txt')).read().strip()
pat = re.compile(r'\d[\d.,]*\s*(?:-|to)?\s*[\d.,]*\s*(?:seconds?|s\b|sec\b|minutes?|min\b|hours?|h\b)', re.I)
ctx = re.compile(r'\b(per|each|/)\b.*\b(abstracts?|citations?|records?|references?|articles?|titles?|documents?)\b|(abstracts?|citations?|records?|references?|articles?|titles?|documents?)\b.*\b(per|each)\b', re.I)
judgement = {
    '42664251': 'LLM processing time per document (Gemini, Llama, Qwen), not human screening time',
    '42458026': 'clinical dysphagia screening time per patient, not literature screening',
    '41378896': 'preload loss within 10 seconds (dental biomechanics), not screening time',
    '40904687': 'LLM processing time per article, not human screening time',
    '40826104': 'manual deduplication time per 100 records and automated filtering time; no human per-record title and abstract screening time',
    '39142995': 'rinse time in seconds (dental materials), not screening time',
    '34407862': 'protocol; no time reported',
    '33437700': 'onset of anaesthesia in seconds, not screening time',
    '25085736': 'time spent on abstract screening was an outcome (dual monitors), but the abstract reports no per-record time (only a difference for data extraction); the full text was not read, per the rule fixed before the search',
}
hits = []
for pmid in es['idlist']:
    p = pubmed[pmid]
    sents = re.split(r'(?<=[.!?])\s+', p['abstract'])
    cand = [s for s in sents if pat.search(s) and ctx.search(s)]
    hits.append({'pmid': pmid, 'title': p['title'], 'journal': p['journal'], 'year': p['year'], 'candidate_sentences': cand,
                 'accepted': False, 'reason': judgement[pmid]})
search = {'database': 'PubMed (esearch, retmax 500)', 'string_as_submitted': qstr, 'query_translation_by_pubmed': es['querytranslation'],
          'count': int(es['count']), 'phrases_dropped_by_pubmed': [x for x in ['"screening of titles and abstracts"', '"per abstract"', '"per citation"'] if x.strip('"') not in es['querytranslation']],
          'run_at': [l.split('\t')[0] for l in open(os.path.join(H, 'fetch_record.tsv')) if '\tpubmed_esearch_screening_time.json\t' in l],
          'hits': hits, 'accepted_sources': [h['pmid'] for h in hits if h['accepted']]}
search['conclusion'] = 'No abstract among the hits reports a measured human per-record title and abstract screening time; under the rule fixed before the search, no conversion of records to reviewer hours is made (R4.1.14).' if not search['accepted_sources'] else 'see accepted_sources'

for r in refs:
    r['all_support_found'] = all(s['exact_match'] or s['alnum_match'] for s in r['support']) if r['support'] else None
out = {'analysis_id': 'AN-0001-14', 'references': refs, 'screening_time_search': search,
       'summary': {r['key']: {'status': r['status'], 'all_support_found': r['all_support_found']} for r in refs}}
json.dump(out, open(os.path.join(H, 'results.json'), 'w'), indent=1, ensure_ascii=False)
with open(os.path.join(H, 'references_verified.csv'), 'w', newline='', encoding='utf8') as f:
    w = csv.writer(f)
    w.writerow(['key', 'doi', 'pmid', 'status', 'all_support_found', 'full_text_reached', 'supports_use', 'nature_reference_draft'])
    for r in refs:
        w.writerow([r['key'], r.get('doi'), r.get('pmid'), r['status'], r['all_support_found'], r.get('full_text_reached'), r.get('supports_use', ''), r['nature_reference_draft']])
for r in refs:
    print(r['key'], r['status'], 'support found:', r['all_support_found'], [(s['exact_match'], s['alnum_match']) for s in r['support']])
print('ref14 abstract identical to Lancet copy:', refs[0]['lancet_copy_identical'])
print('search count', search['count'], 'dropped', search['phrases_dropped_by_pubmed'], 'accepted', search['accepted_sources'])
for h in hits:
    print(' ', h['pmid'], len(h['candidate_sentences']), h['candidate_sentences'][:2])
