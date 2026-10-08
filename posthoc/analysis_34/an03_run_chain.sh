#!/bin/bash
# npj AN-0001-03: hybrid (asr_jevblend_3) with tie-breaking seeds 0-9 on held-out (23) and CLEF (28) reviews, labelling orders saved.
cd /N/project/AiLab/jev/review_pipeline_npj/rounds/round_0001/revision/analysis/AN-0001-03
export PYTHONNOUSERSITE=1
PY=/N/project/AiLab/jev/synergy/.venv/bin/python
date '+%Y-%m-%d %H:%M:%S %Z' > sim_start_time.txt
echo "heldout start $(date '+%Y-%m-%d %H:%M:%S %Z')" >> chain_times.txt
$PY asr_sim_orders_seeds.py /N/project/AiLab/jev/synergy/test_records.json /N/project/AiLab/jev/synergy/jev_test.json an03_hybrid_heldout.json --seeds 10 --procs 22 --methods asr_jevblend_3 --multi-seed-methods asr_jevblend_3 --show > an03_hybrid_heldout.log 2>&1
rc=$?
echo "heldout end $(date '+%Y-%m-%d %H:%M:%S %Z') exit $rc" >> chain_times.txt
echo "clef start $(date '+%Y-%m-%d %H:%M:%S %Z')" >> chain_times.txt
$PY asr_sim_orders_seeds.py /N/project/AiLab/jev/clef/clef_records.json /N/project/AiLab/jev/synergy/jev_clef.json an03_hybrid_clef.json --seeds 10 --procs 22 --methods asr_jevblend_3 --multi-seed-methods asr_jevblend_3 --show > an03_hybrid_clef.log 2>&1
rc=$?
echo "clef end $(date '+%Y-%m-%d %H:%M:%S %Z') exit $rc" >> chain_times.txt
date '+%Y-%m-%d %H:%M:%S %Z' > sim_end_time.txt
echo done > chain.done
