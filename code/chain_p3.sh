#!/bin/bash
cd /N/project/AiLab/jev/synergy
while kill -0 1993346 2>/dev/null; do sleep 20; done
PYTHONNOUSERSITE=1 .venv/bin/python asr_sim.py test_records.json jev_test.json asr_test_ta.json --label ta --seeds 10 --procs 20 --methods asr_prior,asr_jevblend_3 --show > asr_test_ta.log 2>&1
PYTHONNOUSERSITE=1 .venv/bin/python asr_sim.py dev2000_records.json jev_dev2000.json asr_dev_ta.json --label ta --seeds 10 --procs 20 --methods asr_prior,asr_jevblend_3 > asr_dev_ta.log 2>&1
echo done > chain_p3.done
