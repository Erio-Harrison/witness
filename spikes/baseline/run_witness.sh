#!/bin/bash
# The same bench.py against the witness server, with GPU state logged alongside. Run through guarded.sh:
#   ../guarded.sh -m 9G -- ./run_witness.sh LABEL [bench.py args...]
set -u
cd "$(dirname "$0")"
LABEL=${1:?label}; shift
ROOT=../..
PORT=8012
PY=~/.venvs/gpuspike/bin/python
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
    --format=csv,noheader -lms 500 > "gpu_$LABEL.csv" &
SMI=$!
$ROOT/target/release/server $ROOT/data/qwen2.5-coder-1.5b.wit "$(cat model_path.txt)/tokenizer.json" --port $PORT --ctx 5120 ${SERVER_EXTRA:-} > "server_$LABEL.log" 2>&1 &
SRV=$!
trap 'kill $SRV $SMI 2>/dev/null; wait 2>/dev/null' EXIT
for _ in $(seq 120); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break
    kill -0 $SRV 2>/dev/null || { echo "witness server exited during load, see server_$LABEL.log" >&2; exit 1; }
    sleep 0.5
done
# Memory stall over the measured phase: system-wide, and of this run's own processes (its cgroup).
# The second is the one that says whether the benchmark itself was held up.
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.pressure
psi_full() { awk '/^full/{for(i=1;i<=NF;i++) if($i ~ /^total=/){sub("total=","",$i); print $i}}' "$1"; }
p0=$(psi_full /proc/pressure/memory); c0=$(psi_full "$CG")
"$PY" bench.py $PORT "$LABEL" --reps 5 "$@"
rc=$?
# TYPING=1: also the cache-hit first-token test (tools/typing_bench.py), into typing.jsonl.
if [ "${TYPING:-0}" = 1 ]; then "$PY" ../../tools/typing_bench.py "http://127.0.0.1:$PORT" witness "$LABEL" typing.jsonl; fi
echo "BENCH-PHASE memory_full_stall=$(( ($(psi_full /proc/pressure/memory) - p0) / 1000 ))ms own_processes=$(( ($(psi_full "$CG") - c0) / 1000 ))ms" >&2
exit $rc
