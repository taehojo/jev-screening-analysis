#!/bin/bash
cd /N/project/AiLab/jev/synergy
until [ -f chain_clef.done ]; do sleep 60; done
PYTHONNOUSERSITE=1 .venv/bin/python asr_sim.py ../clef/clef_records.json jev_clef.json asr_clef.json --seeds 10 --procs 22 --methods asr_prior,asr_jevblend_3,asr_pseudo_10_50 --show > asr_clef.log 2>&1
PYTHONNOUSERSITE=1 .venv/bin/python clef_eval.py > clef_eval.log 2>&1
echo done > chain_clef2.done
