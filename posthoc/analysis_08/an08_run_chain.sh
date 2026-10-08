#!/bin/bash
# AN-0001-08: re-run of the ASReview (asr_prior, seeds 0-9) and hybrid (asr_jevblend_3) simulations with the labelling orders saved.
cd /N/project/AiLab/jev/review_pipeline/rounds/round_0001/revision/analysis/AN-0001-08
export PYTHONNOUSERSITE=1
PY=/N/project/AiLab/jev/synergy/.venv/bin/python
date > an08_sim_start_time.txt
echo "heldout start $(date)" >> an08_chain_times.txt
$PY asr_sim_orders.py /N/project/AiLab/jev/synergy/test_records.json /N/project/AiLab/jev/synergy/jev_test.json an08_asr_heldout.json --seeds 10 --procs 24 --methods asr_prior,asr_jevblend_3 --show > an08_asr_heldout.log 2>&1
echo "heldout end $(date) exit $?" >> an08_chain_times.txt
echo "clef start $(date)" >> an08_chain_times.txt
$PY asr_sim_orders.py /N/project/AiLab/jev/clef/clef_records.json /N/project/AiLab/jev/synergy/jev_clef.json an08_asr_clef.json --seeds 10 --procs 24 --methods asr_prior,asr_jevblend_3 --show > an08_asr_clef.log 2>&1
echo "clef end $(date) exit $?" >> an08_chain_times.txt
date > an08_sim_end_time.txt
echo done > an08_chain.done
