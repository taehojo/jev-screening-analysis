#!/bin/bash
cd /N/project/AiLab/jev/synergy
until grep -q "chain done" chain_jev.log 2>/dev/null; do sleep 60; done
node run_screen.mjs jev jev_robust_titleonly.json --data dev2000_records.json --criteria criteria_titleonly.json --subset robust_ids.json --batch 10 --conc 3 --gap 1900 > run_jev_titleonly.log 2>&1
node run_screen.mjs jev jev_robust_mismatch.json --data dev2000_records.json --criteria criteria_mismatch.json --subset robust_ids.json --batch 10 --conc 3 --gap 1900 > run_jev_mismatch.log 2>&1
echo "chain2 done $(date)" >> chain2_jev.log
