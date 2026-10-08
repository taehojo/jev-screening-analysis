#!/bin/bash
cd /N/project/AiLab/jev/synergy
while kill -0 1829571 2>/dev/null; do sleep 30; done
PYTHONNOUSERSITE=1 .venv/bin/python zs_baselines.py test --methods tfidf,bm25,minilm,bge --threads 12 > zs_test.log 2>&1
echo "cpu chain done $(date)" >> chain_cpu.log
