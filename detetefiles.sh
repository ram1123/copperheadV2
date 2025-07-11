#!/bin/bash

# find /eos/purdue/store/user/rasharma/*CustomNano*/ -type f ! -name '*_NanoAOD.root'
# replace :
# /eos/purdue -> root://eos.cms.rcac.purdue.edu:1094/
# then run command
# gfal-rm

base="/eos/purdue"
gfal="root://eos.cms.rcac.purdue.edu:1094"

find "$base/store/user/rasharma/"*customNano*/ \
    -type f ! -name '*_NanoAOD.root' \
    -print0 | while IFS= read -r -d '' file; do

  # strip the local prefix and prepend the GFAL endpoint
  url="${gfal}${file#${base}}"
  echo "Processing $url"

  echo "Deleting $url"
  gfal-rm "$url" \
    || echo "  ❌ failed to delete $url" >&2

done
