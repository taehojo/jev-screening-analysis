#!/bin/bash
cd /N/project/AiLab/jev/synergy
until grep -q "^records" ../clef/build_clef.log; do sleep 30; done
# CPU: 무료 기준선 (bge 등)
(PYTHONNOUSERSITE=1 .venv/bin/python zs_baselines.py clef --methods tfidf,bm25,minilm,bge --threads 8 > zs_clef.log 2>&1 &)
# 게이트웨이: 묶음 효과 실행이 끝난 뒤 Jev 채점
until [ -f run_p2.done ]; do sleep 30; done
node run_screen.mjs jev jev_clef.json --data ../clef/clef_records.json --criteria ../clef/clef_criteria.json --batch 10 --conc 3 --gap 1900 > run_jev_clef.log 2>&1
node run_screen.mjs jev jev_clef.json --data ../clef/clef_records.json --criteria ../clef/clef_criteria.json --batch 10 --conc 3 --gap 1900 >> run_jev_clef.log 2>&1
echo done > chain_clef.done
