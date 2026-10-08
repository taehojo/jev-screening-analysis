# Data and code for "Label-free, low-cost AI ranking for systematic review screening"

This archive contains the analysis code, the model inputs, the per-record model responses, and the result files of the study that evaluated the decision model Jev on 169,982 records from 128 systematic reviews (SYNERGY+ development and held-out reviews, CLEF 2019 Cochrane reviews).

## Contents

| Folder | Contents |
|---|---|
| `code/` | Scripts that build the record sets from SYNERGY+ and CLEF 2019 (`build_data.py`, `build_clef.py`), score records with Jev and the generative LLMs (`run_screen.mjs`, `eval_batch.mjs`, probes), compute the free rankers (`zs_baselines.py`, `embed_mxbai.py`), simulate active learning (`asr_sim.py`, `al_sim.py`), evaluate ranking and stopping (`final_eval.py`, `post_eval.py`, `stop_eval.py`, `robust_eval.py`, `clef_eval.py`), and draw figures. Shell scripts show the order in which the runs were chained. |
| `prompts/` | The criteria blocks given to the models for each SYNERGY+ review (`criteria.json`), the mismatched-criteria and title-only controls, and the criteria of the CLEF 2019 reviews with their sources. The question and batch preamble are in `code/run_screen.mjs`. |
| `responses/` | Per-record responses of every model: Jev on the development (`jev_dev2000.json`), held-out (`jev_test.json`), and CLEF (`jev_clef.json`) records; the rephrasings and controls (`jev_robust_*.json`); the 2002-record comparison subset with Jev in two scoring modes and with Claude Opus 5.5, GPT-4o-mini, and DeepSeek-V3.1 (`jev_testcmp_*.json`, `test_cmp_*.json`). Fields: `id` (review, record identifier, index), `review`, `label`, `backend`, `p` (probability), `tokens`, `batch` (records per request), `ok`; LLM files also keep the raw reply. For the composition of the ten-record requests, see post-hoc analysis 22. |
| `results/` | Result files of the planned analyses (ranking, active learning, stopping, CLEF) and the free-ranker scores (`scores_zs/`). |
| `posthoc/analysis_NN/` | Scripts and machine-readable outputs of the post-hoc analyses 1 to 52, numbered as in Supplementary Tables 1 to 3; `posthoc/INDEX.csv` lists them. |

## Not included

- Titles and abstracts of the records. They can be rebuilt from SYNERGY+ (release synergy_plus_v3.0) and the CLEF 2019 collection with `code/build_data.py` and `code/build_clef.py`. In five result files, `abstract` fields were replaced by `"[removed]"`.
- All records, screening decisions, and the criteria block of the pilot review, which is unpublished; they will be available from the corresponding author after the publication of that review. Post-hoc analyses 16, 21, and 26 are not included for the reasons given in `posthoc/INDEX.csv`.
- The analysis plan file and the memo of 25 September 2026.
- The session records of the AI coding tool, process notes and logs of the analyses, and third-party documents that were downloaded to verify references.

An e-mail address in two files was replaced by `[email withheld]`. The scripts keep the absolute paths of the analysis server.

## Reproduction

The Jev endpoint reports no model version, so re-querying the model may not reproduce these scores; the files in `responses/` are the record of the primary analysis. The threshold and statistical-stopping results of Table 3 can be reproduced with `validation/reproduce_paper.py` in ReviewFast (https://doi.org/10.5281/zenodo.23243074).

## License

Code: Apache License 2.0 (`LICENSE`). Model responses and result files produced in this study: CC BY 4.0. Screening labels come from SYNERGY+ and CLEF 2019 and remain under the terms of those sources.
