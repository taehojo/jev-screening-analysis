#!/bin/bash
# AN-0001-03: lambda = 5 and 10 sensitivity of the hybrid (unchanged copies of synergy/asr_sim.py and al_sim.py; output paths in this folder).
cd /N/project/AiLab/jev/review_pipeline/rounds/round_0001/revision/analysis/AN-0001-03
export PYTHONNOUSERSITE=1
PY=/N/project/AiLab/jev/synergy/.venv/bin/python
date > an03_start_time.txt
echo "dev start $(date)" >> an03_chain_times.txt
$PY asr_sim.py /N/project/AiLab/jev/synergy/dev2000_records.json /N/project/AiLab/jev/synergy/jev_dev2000.json an03_asr_dev.json --seeds 1 --procs 24 --methods asr_jevblend_5,asr_jevblend_10 --show > an03_asr_dev.log 2>&1
echo "dev end $(date) exit $?" >> an03_chain_times.txt
echo "heldout start $(date)" >> an03_chain_times.txt
$PY asr_sim.py /N/project/AiLab/jev/synergy/test_records.json /N/project/AiLab/jev/synergy/jev_test.json an03_asr_heldout.json --seeds 1 --procs 24 --methods asr_jevblend_5,asr_jevblend_10 --show > an03_asr_heldout.log 2>&1
echo "heldout end $(date) exit $?" >> an03_chain_times.txt
echo "clef start $(date)" >> an03_chain_times.txt
$PY asr_sim.py /N/project/AiLab/jev/clef/clef_records.json /N/project/AiLab/jev/synergy/jev_clef.json an03_asr_clef.json --seeds 1 --procs 24 --methods asr_jevblend_5,asr_jevblend_10 --show > an03_asr_clef.log 2>&1
echo "clef end $(date) exit $?" >> an03_chain_times.txt
date > an03_end_time.txt
echo done > an03_chain.done
