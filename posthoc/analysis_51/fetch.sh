#!/bin/bash
# AN-0003-01: fetch one public URL with curl, save the raw response in sources/.
# Copied from rounds/round_0002/revision/analysis/AN-0002-04/fetch.sh (itself a copy of AN-0001-14/fetch.sh). Difference from AN-0002-04: folder path only. Differences of AN-0002-04 from AN-0001-14: folder path;
# a line with status PENDING is appended to fetch_record.tsv before the request is sent
# (plan: every request is logged before it is sent), and a second line with the result after it.
# usage: fetch.sh NAME URL [ACCEPT_HEADER]
D=/N/project/AiLab/jev/review_pipeline_npj/rounds/round_0003/revision/analysis/AN-0003-01
S=$D/sources; N="$1"; U="$2"; A="${3:-*/*}"
[ -f $D/fetch_record.tsv ] || printf 'time\tname\tstatus\tbytes\tcontent_type\turl\tfinal_url\n' > $D/fetch_record.tsv
T=$(date '+%Y-%m-%d %H:%M:%S %Z')
printf '%s\t%s\tPENDING\t\t\t%s\t\n' "$T" "$N" "$U" >> $D/fetch_record.tsv
R=$(curl -sL --max-time 60 -A "Mozilla/5.0 (research reference verification)" -H "Accept: $A" -o "$S/$N" -w '%{http_code}\t%{size_download}\t%{content_type}\t%{url_effective}' "$U")
T2=$(date '+%Y-%m-%d %H:%M:%S %Z')
printf '%s\t%s\t%s\t%s\n' "$T2" "$N" "$(echo "$R" | cut -f1-3)" "$U	$(echo "$R" | cut -f4)" >> $D/fetch_record.tsv
echo "$N $(echo "$R" | cut -f1-2)"
sleep 0.5
