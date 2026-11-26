#!/bin/bash
# llama.cpp baseline with GPU state logged alongside. Run through guarded.sh:
#   ../guarded.sh -- ./run_llamacpp.sh LABEL [bench.py args...]
# Writes gpu_LABEL.csv (clocks, power, temperature every 500 ms) and server_LABEL.log; bench.py
# appends its rows to results.jsonl under LABEL.
set -u
cd "$(dirname "$0")"
LABEL=${1:?label}; shift
PORT=8300
SERVER=~/gpu-spike/llama.cpp/build/bin/llama-server
MODEL=~/gpu-spike/models/qwen2.5-coder-1.5b-f16.gguf
PY=~/.venvs/gpuspike/bin/python

nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
    --format=csv,noheader -lms 500 > "gpu_$LABEL.csv" &
SMI=$!
"$SERVER" -m "$MODEL" -np 4 -c 20480 -ngl 99 --cache-ram 1024 --host 127.0.0.1 --port $PORT > "server_$LABEL.log" 2>&1 &
SRV=$!
trap 'kill $SRV $SMI 2>/dev/null; wait 2>/dev/null' EXIT

for _ in $(seq 120); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break
    kill -0 $SRV 2>/dev/null || { echo "llama-server exited during load" >&2; exit 1; }
    sleep 0.5
done
"$PY" bench.py $PORT "$LABEL" --reps 5 "$@"
