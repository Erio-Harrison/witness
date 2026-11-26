#!/bin/bash
# Decode / prefill timing of witness next to the matmul-only prototype (spikes/baseline/gemv), run
# alternately in one session so that both see the same GPU state.
#
#   tools/perf.sh [ROUNDS] [bench args...]        env: BENCH_LENS, WK_SKIP are passed through
#
# Prints, per round, the prototype's whole-step time for 1 and 4 rows, then the bench lines, then
# the GPU clocks and power sampled while the bench ran.
cd "$(dirname "$0")/.."
ROUNDS=${1:-3}; shift
MODEL=data/qwen2.5-coder-1.5b.wit
TOKENS=data/ref-1.5b/fim/tokens.i32
for i in $(seq "$ROUNDS"); do
    proto=$(spikes/baseline/gemv | awk '/B=[14] ours/{printf "B=%s %s ms  ", substr($1,3), $4}')
    nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader,nounits -lms 500 > /tmp/witness_perf_gpu.$$ &
    smi=$!
    out=$(./target/release/bench $MODEL $TOKENS --reps 3 "$@" 2>&1 | grep "^prompt")
    kill $smi; wait $smi 2>/dev/null
    gpu=$(awk -F', ' '{sm+=$1; mem+=$2; w+=$3; if($4>t)t=$4; n++} END{printf "SM %.0f MHz, memory %.0f MHz, %.1f W, max %d C", sm/n, mem/n, w/n, t}' /tmp/witness_perf_gpu.$$)
    rm -f /tmp/witness_perf_gpu.$$
    echo "round $i  prototype: $proto"
    echo "$out" | sed 's/^/    /'
    echo "    gpu: $gpu"
done
