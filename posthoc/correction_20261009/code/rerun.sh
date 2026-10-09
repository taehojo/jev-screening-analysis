#!/bin/bash
# Re-run the pilot-review analyses with the corrected labels (9 October 2026).
# orig_*: unchanged copies of the analysis scripts (check that the stored results are reproduced).
# new_*:  the same copies with the pilot input paths pointed to relabel_20261009/data/ (corrected labels).
set -e
J=/N/project/AiLab/jev
W=$J/relabel_20261009
PY=$J/synergy/.venv/bin/python
A=$J/review_pipeline/rounds/round_0001/revision/analysis
R=$W/rerun
mkdir -p $R
cp $J/screen/subset544.json $W/data/subset544.json
date "+start %F %T %Z"

# free rankers (needed by analysis 10)
$PY $W/zs_auc.py $R/zs_auc_dhl.csv

# active learning on the pilot review (needed by analysis 2): same call as the stored asr_dhl.json
cd $J/synergy
$PY asr_sim.py dhl_records.json dhl_jev_b10.json $R/asr_dhl_orig.json --methods asr_prior,asr_random,asr_jev,asr_jevblend_3 --seeds 10 --procs 22 > $R/asr_dhl_orig.log 2>&1
$PY asr_sim.py $W/data/dhl_records.json dhl_jev_b10.json $R/asr_dhl.json --methods asr_prior,asr_random,asr_jev,asr_jevblend_3 --seeds 10 --procs 22 > $R/asr_dhl.log 2>&1

for an in AN-0001-01:calibration.py AN-0001-06:ta_stopping.py AN-0001-10:an10_year_strata.py AN-0001-02:jev_only_ranking.py; do
  d=${an%%:*}; f=${an##*:}
  for v in orig new; do
    mkdir -p $R/${v}_$d; cp $A/$d/$f $R/${v}_$d/
    [ "$d" = "AN-0001-02" ] && cp $A/$d/al_sim_copy.py $R/${v}_$d/
  done
  N=$R/new_$d/$f
  sed -i \
    -e "s#{ROOT}/synergy/dhl_records.json#{ROOT}/relabel_20261009/data/dhl_records.json#g" \
    -e "s#{ROOT}/synergy/dhl_jev_b10.json#{ROOT}/relabel_20261009/data/dhl_jev_b10.json#g" \
    -e "s#{ROOT}/screen/screen_jev.json#{ROOT}/relabel_20261009/data/screen_jev.json#g" \
    -e "s#S + 'dhl_jev_b10.json'#'$W/data/dhl_jev_b10.json'#g" \
    -e "s#S + 'zs_auc_dhl.csv'#'$R/zs_auc_dhl.csv'#g" \
    -e "s#sc = '/N/project/AiLab/jev/screen/'#sc = '$W/data/'#g" \
    -e "s#SYN + '/dhl_records.json'#'$W/data/dhl_records.json'#g" \
    -e "s#SYN + '/dhl_jev_b10.json'#'$W/data/dhl_jev_b10.json'#g" \
    -e "s#SYN + '/asr_dhl.json'#'$R/asr_dhl.json'#g" \
    -e "s#PB.y_ta.sum() == 144 and PB.y.sum() == 77 and PS.y_ta.sum() == 144 and PS.y.sum() == 77#PB.y_ta.sum() == 143 and PB.y.sum() == 76 and PS.y_ta.sum() == 143 and PS.y.sum() == 76#" \
    $N
  diff $A/$d/$f $N > $R/new_$d/PATCH.diff || true
  for v in orig new; do
    (cd $R/${v}_$d && $PY $f > run.log 2>&1) || echo "FAILED $v $d"
  done
done
date "+end %F %T %Z"
