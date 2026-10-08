#!/bin/bash
# AN-0002-03: one more read-only re-query of the Internet Archive CDX index of the reference 16 page at about
# 22:30 EDT (REVISION_ANALYSIS_LOG.md entry of 21:03:04 EDT). Output: cdx_blog_post_requery_2230.json and a line in
# an03_requests_attempt2.log. No Save Page Now request.
cd /N/project/AiLab/jev/review_pipeline/rounds/round_0002/revision/analysis/AN-0002-03
while [ "$(date +%H%M)" -lt 2230 ]; do sleep 30; done
for i in 1 2 3; do
  T=$(date '+%Y-%m-%d %H:%M:%S %Z')
  code=$(curl -sS --max-time 120 -o cdx_blog_post_requery_2230.json -w "%{http_code}" "https://web.archive.org/cdx/search/cdx?url=typesafe.ai/blog/introducing-system-one-models-and-jev&output=json&fl=timestamp,statuscode,digest,mimetype,original" 2>&1)
  echo "$T curl CDX re-query (22:30 retry $i) -> $code" >> an03_requests_attempt2.log
  [ "$code" = "200" ] && break
  sleep 120
done
echo "done $(date '+%Y-%m-%d %H:%M:%S %Z')" >> an03_requests_attempt2.log
