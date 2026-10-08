#!/bin/bash
# AN-0001-15: fetch the vendor's public pages (blog post cited as reference 16, site pages, documentation, pricing, terms, privacy,
# company/about) and the gateway's public model page and listing, on a recorded date. No authorisation header, no study text.
cd /N/project/AiLab/jev/review_pipeline/rounds/round_0001/revision/analysis/AN-0001-15/sources
UA="Mozilla/5.0"
rec=fetch_record.txt
echo "Fetched with curl -sS -L -A \"$UA\" from this server; start $(date)" > $rec
get() { # $1 = output file, $2 = url
  code=$(curl -sS -L -A "$UA" -m 60 -o "$1" -w "%{http_code}" "$2" 2>>"$rec"); echo "$1 <- $2 HTTP $code size $(stat -c %s "$1" 2>/dev/null) $(date '+%H:%M:%S %Z')" >> $rec; }
get typesafe_blog_introducing_system_one_models_and_jev.html "https://typesafe.ai/blog/introducing-system-one-models-and-jev"
get typesafe_home.html "https://typesafe.ai/"
get typesafe_blog_index.html "https://typesafe.ai/blog"
get typesafe_docs.html "https://typesafe.ai/docs"
get typesafe_pricing.html "https://typesafe.ai/pricing"
get typesafe_terms.html "https://typesafe.ai/terms"
get typesafe_privacy.html "https://typesafe.ai/privacy"
get typesafe_about.html "https://typesafe.ai/about"
get vercel_model_page_jev.html "https://vercel.com/ai-gateway/models/jev"
get vercel_ai_gateway_v1_models.json "https://ai-gateway.vercel.sh/v1/models"
get vercel_ai_gateway_docs_models.html "https://vercel.com/docs/ai-gateway/models-and-providers"
echo "end $(date)" >> $rec
# second stage: any typesafe.ai links found in the fetched pages that were not fetched above
grep -ohE 'https?://(www\.)?typesafe\.ai[^"'"'"' <>)]*' typesafe_*.html 2>/dev/null | sed 's/[#?].*$//' | sort -u > typesafe_links_found.txt
echo "links found on typesafe.ai pages: $(wc -l < typesafe_links_found.txt)" >> $rec
