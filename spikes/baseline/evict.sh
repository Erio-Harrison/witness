#!/bin/bash
# Drop this project's large generated files from the page cache (no root needed), so that a benchmark
# does not start with the cache full and stall on reclaim. They are read again on next use.
cd "$(dirname "$0")/../.."
for f in data/ref-*/*/*.f32 data/qwen2.5-0.5b-instruct.wit; do
    [ -f "$f" ] && dd if="$f" iflag=nocache count=0 status=none
done
free -m | awk 'NR==2{print "page cache now " $6 " MB, free " $4 " MB"}'
