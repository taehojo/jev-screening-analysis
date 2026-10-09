#!/bin/bash
# Re-run analysis 32 on the corrected analysis 6, then stage the deposit folder posthoc/correction_20261009.
# Only aggregated results and code are staged; no record of the pilot review, no record identifier, no title.
set -e
J=/N/project/AiLab/jev
W=$J/relabel_20261009
R=$W/rerun
PY=$J/synergy/.venv/bin/python
A32=$J/review_pipeline_npj/rounds/round_0001/revision/analysis/AN-0001-01

for v in orig new; do mkdir -p $R/${v}_AN32; cp $A32/an01_derived.py $R/${v}_AN32/; done
sed -i "s#R6 = json.load(open(AN + 'AN-0001-06/results.json'))#R6 = json.load(open('$R/new_AN-0001-06/results.json'))#" $R/new_AN32/an01_derived.py
diff $A32/an01_derived.py $R/new_AN32/an01_derived.py > $R/new_AN32/PATCH.diff || true
for v in orig new; do (cd $R/${v}_AN32 && $PY an01_derived.py > run.log 2>&1); done
$PY - <<EOF
import json
def strip(o):
    if isinstance(o, dict): return {k: strip(v) for k, v in o.items() if k not in ("finished",)}
    if isinstance(o, list): return [strip(v) for v in o]
    return o
s = strip(json.load(open("$A32/results.json"))); o = strip(json.load(open("$R/orig_AN32/results.json"))); n = strip(json.load(open("$R/new_AN32/results.json")))
print("analysis 32: stored == orig:", s == o)
print("analysis 32 pilot row, orig:", o["b_crosstab"]["pilot_aggregated_batched"])
print("analysis 32 pilot row, new: ", n["b_crosstab"]["pilot_aggregated_batched"])
o.pop("b_crosstab"); n.pop("b_crosstab"); print("analysis 32: outside b_crosstab unchanged:", o == n)
EOF

S=$J/deposit_zenodo/correction_20261009
rm -rf $S; mkdir -p $S/code
cp $W/rerun.sh $W/rerun_32.sh $W/compare.py $W/zs_auc.py $S/code/ 2>/dev/null || cp $W/rerun.sh $W/compare.py $W/zs_auc.py $S/code/
cp $R/compare_report.json $S/
for pair in "AN-0001-01:analysis_01:results.json summary_table.csv" "AN-0001-02:analysis_02:results.json per_review.csv jev_only_per_seed.csv" \
            "AN-0001-06:analysis_06:results.json per_review_tau_ta.csv" "AN-0001-10:analysis_10:results.json"; do
  src=${pair%%:*}; rest=${pair#*:}; dst=${rest%%:*}; files=${rest#*:}
  mkdir -p $S/$dst
  for f in $files PATCH.diff; do cp $R/new_$src/$f $S/$dst/; done
done
mkdir -p $S/analysis_32; cp $R/new_AN32/results.json $R/new_AN32/PATCH.diff $S/analysis_32/
cp $W/stage_correction.sh $S/code/
sed -i "s/[email withheld]/[email withheld]/g" $(grep -rl "[email withheld]" $S || true) 2>/dev/null || true
find $S -type f | sort
