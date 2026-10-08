#!/bin/bash
cd /N/project/AiLab/jev/synergy
until grep -q "^done" embed_mxbai_test.log; do sleep 60; done
PYTHONNOUSERSITE=1 .venv/bin/python asr_sim.py test_records.json jev_test.json asr_test.json --seeds 10 --procs 20 --methods h3_prior,h3_jevblend_3 --show > asr_test_p4.log 2>&1
echo done > chain_p4.done
