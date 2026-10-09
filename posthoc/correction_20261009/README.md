# Correction of the pilot-review labels (9 October 2026)

The records and screening decisions of the pilot review are not deposited (see the README at the top of the archive). This folder holds the aggregated results of the post-hoc analyses that used the pilot labels, recomputed after a correction of those labels, and the code of the re-runs.

## What was corrected

The PubMed records of the pilot review had been matched by DOI and title to the records that passed title and abstract screening. The script that downloaded the PubMed records stored, for some records, the DOI of a cited reference instead of the DOI of the record itself, and the matching by DOI mislabeled three records: two records that had not passed title and abstract screening were counted as positive, one of them as an included study, and one record that had passed was counted as negative. The matching was redone by title. Of the 1572 PubMed records, 143 passed title and abstract screening (previously 144) and 76 were finally included (previously 77). The included study that the first matching had assigned to another record was scored by Jev with a single-record request on 9 October 2026; the conclusion-level analysis (analysis 16, not deposited) gave the same results with that score.

## Contents

| Path | Contents |
|---|---|
| `analysis_01/`, `analysis_02/`, `analysis_06/`, `analysis_10/`, `analysis_32/` | Result files of the post-hoc analyses with the same numbers (`posthoc/analysis_NN`), recomputed with the corrected labels. `PATCH.diff` shows the only changes to each script: the input paths of the pilot files and, in analysis 1, the expected counts. Analysis 32 takes its pilot row from analysis 6. The result files in `posthoc/analysis_NN` are unchanged and use the labels before the correction. |
| `compare_report.json` | Comparison of the stored results with re-runs of the unchanged scripts on the original inputs, which were identical apart from four fixed-effects recalibration slopes of analysis 1 for data sets other than the pilot review (differences below 1e-6), and with the re-runs on the corrected inputs, which differed only in pilot entries. |
| `code/` | `rerun.sh`, `compare.py`, `zs_auc.py`, and `stage_correction.sh`. They read pilot files that are not deposited. |

Only the pilot entries of these result files changed. The values reported in the Supplementary Information are those of this folder.
