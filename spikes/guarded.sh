#!/bin/bash
# Run an experiment so that running out of memory cannot take the WSL session down with it, and so
# that a run disturbed by memory pressure is flagged instead of silently kept.
#
#   guarded.sh [-m MEMMAX] -- CMD ARGS...      (default MEMMAX 6G)
#
# - CMD runs in its own transient systemd user scope with MemoryMax and OOMPolicy=continue: an OOM
#   inside it kills only its own processes, not init.scope (VS Code server, Claude Code).
# - Before/after it compares the kernel's OOM kill counter and memory PSI "full" stall time, and
#   prints GUARD-VERDICT: clean | DIRTY (...). A DIRTY run's numbers must not be kept.
set -u
MEM=6G
if [ "${1:-}" = "-m" ]; then MEM=$2; shift 2; fi
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "usage: guarded.sh [-m MEMMAX] -- CMD..." >&2; exit 2; }

oom_kills() { awk '/^oom_kill /{print $2}' /proc/vmstat; }
psi_full() { awk '/^full/{for(i=1;i<=NF;i++) if($i ~ /^total=/){sub("total=","",$i); print $i}}' /proc/pressure/memory; }

k0=$(oom_kills); p0=$(psi_full); t0=$(date +%s.%N)
systemd-run --user --scope --quiet -p MemoryMax="$MEM" -p MemorySwapMax=0 -p OOMPolicy=continue -- "$@"
rc=$?
k1=$(oom_kills); p1=$(psi_full); t1=$(date +%s.%N)

stall_ms=$(( (p1 - p0) / 1000 ))
reasons=()
[ "$rc" -ne 0 ] && reasons+=("exit=$rc")
[ "$k1" -gt "$k0" ] && reasons+=("oom_kills=$((k1 - k0))")
[ "$stall_ms" -gt 50 ] && reasons+=("memory_full_stall=${stall_ms}ms")
if [ ${#reasons[@]} -eq 0 ]; then
    echo "GUARD-VERDICT: clean (mem cap $MEM, $(printf '%.0f' "$(echo "$t1 - $t0" | bc)")s)" >&2
else
    echo "GUARD-VERDICT: DIRTY (${reasons[*]})" >&2
fi
exit $rc
