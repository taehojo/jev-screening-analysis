#!/bin/bash
cd /N/project/AiLab/jev/synergy
until [ -f chain_p3.done ]; do sleep 20; done
PYTHONNOUSERSITE=1 .venv/bin/python asr_sim.py dev2000_records.json jev_dev2000.json asr_dev_p1.json --seeds 1 --procs 20 --methods asr_pseudo_1_50,asr_pseudo_1_80,asr_pseudo_2_50,asr_pseudo_2_80,asr_pseudo_5_50,asr_pseudo_5_80,asr_pseudo_10_50,asr_pseudo_10_80 > asr_dev_p1.log 2>&1
echo done > chain_p1.done
