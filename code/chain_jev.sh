#!/bin/bash
# Jev 작업을 게이트웨이 한도 안에서 순서대로 실행한다
cd /N/project/AiLab/jev/synergy
while kill -0 1801949 2>/dev/null; do sleep 30; done
node run_screen.mjs jev jev_dev2000.json --data dev2000_records.json --criteria criteria.json --batch 10 --conc 3 --gap 1900 >> run_jev_dev.log 2>&1   # 실패분 재시도
node run_screen.mjs jev jev_test.json --data test_records.json --criteria criteria.json --batch 10 --conc 3 --gap 1900 > run_jev_test.log 2>&1
node run_screen.mjs jev jev_test.json --data test_records.json --criteria criteria.json --batch 10 --conc 3 --gap 1900 >> run_jev_test.log 2>&1   # 실패분 재시도
node run_screen.mjs jev jev_robust_q1.json --data dev2000_records.json --criteria criteria.json --subset robust_ids.json --q q1 --batch 10 --conc 3 --gap 1900 > run_jev_q1.log 2>&1
node run_screen.mjs jev jev_robust_q2.json --data dev2000_records.json --criteria criteria.json --subset robust_ids.json --q q2 --batch 10 --conc 3 --gap 1900 > run_jev_q2.log 2>&1
echo "chain done $(date)" >> chain_jev.log
