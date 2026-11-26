#!/bin/bash
# Baseline of one comparison framework with GPU state logged alongside. Run through guarded.sh:
#   ../guarded.sh -m 9G -- ./run_baseline.sh vllm|sglang|trtllm LABEL [bench.py args...]
# Same model weights (HF snapshot in model_path.txt, bf16) and same bench.py for every framework.
# Writes gpu_LABEL.csv (clocks, power, temperature every 500 ms) and server_LABEL.log; bench.py
# appends its rows to results.jsonl under LABEL.
set -u
cd "$(dirname "$0")"
FW=${1:?framework}; LABEL=${2:?label}; shift 2
MODEL=$(cat model_path.txt)
PY=~/.venvs/gpuspike/bin/python
CTX=5120   # 4096 prompt + 128 generated, same per-slot context as the llama.cpp run

case $FW in
vllm)
    PORT=8301
    CMD=(~/.venvs/vllm/bin/vllm serve "$MODEL" --dtype bfloat16 --max-model-len $CTX
         --gpu-memory-utilization 0.85 --host 127.0.0.1 --port $PORT) ;;
sglang)
    PORT=8302
    export PATH=~/.venvs/sglang/bin:$PATH   # flashinfer builds its kernels with the venv's ninja at first use
    CMD=(~/.venvs/sglang/bin/python -m sglang.launch_server --model-path "$MODEL" --dtype bfloat16
         --context-length $CTX --mem-fraction-static 0.8 --host 127.0.0.1 --port $PORT) ;;
trtllm)
    PORT=8303
    export PATH=~/.venvs/trtllm/bin:$PATH
    # Its worker is spawned over Open MPI; on WSL the TCP transport picks the wrong interface and the
    # worker never connects. Keep MPI on loopback.
    export OMPI_MCA_btl_tcp_if_include=lo OMPI_MCA_oob_tcp_if_include=lo
    CMD=(~/.venvs/trtllm/bin/trtllm-serve serve "$MODEL" --max_seq_len $CTX --max_batch_size 4
         --free_gpu_memory_fraction 0.5 --no-telemetry --host 127.0.0.1 --port $PORT) ;;
*) echo "unknown framework $FW" >&2; exit 2 ;;
esac

nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
    --format=csv,noheader -lms 500 > "gpu_$LABEL.csv" &
SMI=$!
# SERVER_EXTRA: more server arguments (split on spaces), e.g. to turn speculative decoding on.
setsid "${CMD[@]}" ${SERVER_EXTRA:-} > "server_$LABEL.log" 2>&1 &
SRV=$!
trap 'kill -- -$SRV $SMI 2>/dev/null; wait 2>/dev/null' EXIT

for _ in $(seq 600); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break
    kill -0 $SRV 2>/dev/null || { echo "$FW server exited during load, see server_$LABEL.log" >&2; exit 1; }
    sleep 0.5
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "$FW server not ready after 300 s" >&2; exit 1; }
# guarded.sh's verdict covers the whole run including model load; this is the measured part alone.
# Memory stall over the measured phase: system-wide, and of this run's own processes (its cgroup).
# The second is the one that says whether the benchmark itself was held up.
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.pressure
psi_full() { awk '/^full/{for(i=1;i<=NF;i++) if($i ~ /^total=/){sub("total=","",$i); print $i}}' "$1"; }
p0=$(psi_full /proc/pressure/memory); c0=$(psi_full "$CG")
"$PY" bench.py $PORT "$LABEL" --reps 5 "$@"
rc=$?
# TYPING=1: also the cache-hit first-token test (tools/typing_bench.py), into typing.jsonl.
if [ "${TYPING:-0}" = 1 ]; then "$PY" ../../tools/typing_bench.py "http://127.0.0.1:$PORT" openai "$LABEL" typing.jsonl; fi
echo "BENCH-PHASE memory_full_stall=$(( ($(psi_full /proc/pressure/memory) - p0) / 1000 ))ms own_processes=$(( ($(psi_full "$CG") - c0) / 1000 ))ms" >&2
exit $rc
