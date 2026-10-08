#!/bin/bash
# AN-0001-14: fetch one URL with curl, save the raw response in sources/, append a line to fetch_record.tsv.
# usage: fetch.sh NAME URL [ACCEPT_HEADER]
D=/N/project/AiLab/jev/review_pipeline_npj/rounds/round_0001/revision/analysis/AN-0001-14
S=$D/sources; N="$1"; U="$2"; A="${3:-*/*}"
[ -f $D/fetch_record.tsv ] || printf 'time\tname\tstatus\tbytes\tcontent_type\turl\tfinal_url\n' > $D/fetch_record.tsv
T=$(date '+%Y-%m-%d %H:%M:%S %Z')
R=$(curl -sL --max-time 60 -A "Mozilla/5.0 (research reference verification)" -H "Accept: $A" -o "$S/$N" -w '%{http_code}\t%{size_download}\t%{content_type}\t%{url_effective}' "$U")
printf '%s\t%s\t%s\t%s\n' "$T" "$N" "$(echo "$R" | cut -f1-3)" "$U	$(echo "$R" | cut -f4)" >> $D/fetch_record.tsv
echo "$N $(echo "$R" | cut -f1-2)"
sleep 0.5
