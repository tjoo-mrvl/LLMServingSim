#!/bin/bash
# -----------------------------------------------------------------------------
# Acquire one native MoE deployment's components, one process per GPU.
#
# Run from inside the vLLM Docker (scripts/docker-vllm.sh) at /workspace:
#
#     MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 HARDWARE=RTXPRO6000 TP=2 \
#         ./profiler/profile-moe-ranks.sh
#
# The target is TP x DP with EP = TP * DP, vLLM's full expert parallelism.
# Each EP rank is measured by `slice --group moe --moe-ranks` on one physical
# GPU. Every visible GPU (CUDA_VISIBLE_DEVICES, else every GPU nvidia-smi
# lists) runs one process; with G GPUs and more ranks than that, GPU i measures
# ranks i, i+G, i+2G, ... in turn, so a single GPU measures them all. Each rank
# writes only its own table and meta.yaml entry, so the processes share one
# output root, and the one that publishes the last rank logs the rank
# cross-check. Do not run two of these processes on one GPU: they disturb each
# other's timings, and every point is published as measured.
#
# Arguments after the script name are passed to every slice unchanged, e.g.
# --max-num-batched-tokens 4096 or --variant <label>. Ordinary categories are
# untouched; profile.sh measures those.
#
# Output: <OUT_ROOT>/<HARDWARE>/<MODEL>/<variant>/tp<TP>/moe_components/
# Logs:   <LOG_DIR>/moe-rank<ranks>.log, one per process.
# See https://llmservingsim.ai/docs/profiler/native-moe-components.
# -----------------------------------------------------------------------------

set -euo pipefail

MODEL="${MODEL:?set MODEL, e.g. MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507}"
HARDWARE="${HARDWARE:?set HARDWARE, the folder name under perf/}"
TP="${TP:?set TP, the target tensor-parallel degree}"
DP="${DP:-1}"
OUT_ROOT="${OUT_ROOT:-}"                # --out-root; unset writes profiler/perf/
MOE_ROUNDS="${MOE_ROUNDS:-}"            # --moe-rounds; unset keeps the CLI default
MOE_GRID_FACTOR="${MOE_GRID_FACTOR:-}"  # --moe-grid-factor; unset keeps the CLI default
LOG_DIR="${LOG_DIR:-outputs/profile-moe-ranks}"

for name in TP DP; do
    if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $name must be a positive integer, got '${!name}'" >&2
        exit 2
    fi
done
EP="${EP:-$(( TP * DP ))}"
# --ep only selects a TP x DP target; it never changes how experts are sharded.
if [[ "$EP" != "$(( TP * DP ))" ]]; then
    echo "ERROR: EP=$EP must equal TP*DP=$(( TP * DP ))" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

gpus=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=, read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
if (( ${#gpus[@]} == 0 )); then
    echo "ERROR: no visible GPU (CUDA_VISIBLE_DEVICES or nvidia-smi)" >&2
    exit 1
fi
procs=$(( ${#gpus[@]} < EP ? ${#gpus[@]} : EP ))

# The slice refreshes TP, and the CLI wants TP1 in the --tp list too.
tp_list=1
if (( TP > 1 )); then
    tp_list="1,$TP"
fi

mkdir -p "$LOG_DIR"
pids=()
logs=()
for (( p = 0; p < procs; p++ )); do
    ranks=()
    for (( r = p; r < EP; r += procs )); do
        ranks+=("$r")
    done
    rank_list="$(IFS=,; echo "${ranks[*]}")"
    log="$LOG_DIR/moe-rank$(IFS=_; echo "${ranks[*]}").log"
    cmd=(python3 -m profiler slice "$MODEL" --hardware "$HARDWARE"
         --tp "$tp_list" --tp-refresh "$TP" --group moe
         --dp "$DP" --ep "$EP" --moe-ranks "$rank_list")
    [[ -n "$OUT_ROOT" ]]        && cmd+=(--out-root "$OUT_ROOT")
    [[ -n "$MOE_ROUNDS" ]]      && cmd+=(--moe-rounds "$MOE_ROUNDS")
    [[ -n "$MOE_GRID_FACTOR" ]] && cmd+=(--moe-grid-factor "$MOE_GRID_FACTOR")
    cmd+=("$@")
    echo ">>> EP ranks $rank_list on GPU ${gpus[$p]}: $log"
    ( export CUDA_VISIBLE_DEVICES="${gpus[$p]}"
      echo "+ ${cmd[*]}"
      "${cmd[@]}"
    ) > "$log" 2>&1 &
    pids+=("$!")
    logs+=("$log")
done

failed=()
for (( p = 0; p < procs; p++ )); do
    wait "${pids[$p]}" || failed+=("${logs[$p]}")
done
if (( ${#failed[@]} )); then
    echo "ERROR: these processes failed: ${failed[*]}" >&2
    exit 1
fi
echo ">>> TP=$TP DP=$DP EP=$EP: all $EP ranks measured under" \
     "${OUT_ROOT:-profiler/perf}/$HARDWARE/$MODEL/<variant>/tp$TP/moe_components/"
echo ">>> The log that finished last ends with the rank cross-check."
