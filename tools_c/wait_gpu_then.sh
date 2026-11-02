#!/usr/bin/env bash
# Wait, then run a command once the GPU has no compute process (two checks 2 min apart). Polls every 10 min until a deadline.
#   bash tools_c/wait_gpu_then.sh <start_delay_s> <max_wait_after_start_s> <command...>
set -u
DELAY=$1; MAXW=$2; shift 2
echo "[$(date '+%F %T')] waiting ${DELAY}s before the first GPU check"
sleep "$DELAY"
END=$(( $(date +%s) + MAXW ))
busy() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }
while :; do
    if ! busy; then
        sleep 120
        if ! busy; then
            echo "[$(date '+%F %T')] GPU free -> $*"
            "$@"
            echo "[$(date '+%F %T')] exit $?"
            exit 0
        fi
    fi
    echo "[$(date '+%F %T')] GPU busy: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ' ')"
    if [ "$(date +%s)" -ge "$END" ]; then
        echo "[$(date '+%F %T')] gave up: GPU still busy"
        exit 1
    fi
    sleep 600
done
