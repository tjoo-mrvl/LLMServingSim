#!/usr/bin/env bash
# =============================================================================
# profile-deepseek-h200.sh — DeepSeek-R1 / V4-Flash layerwise profiler driver
# -----------------------------------------------------------------------------
# Run INSIDE the vLLM/profiler container (scripts/docker-vllm.sh, vLLM 0.31),
# from the repo root. Each profiler process uses ONE GPU + dummy weights
# (load_format=dummy); STAGE=moe runs eight such processes, one per GPU. It
# reads the vendored configs/model/deepseek-ai/*.json and downloads NO
# weights. R1 output lands in profiler/perf/H200/deepseek-ai/DeepSeek-R1/fp8/:
# the variant follows the checkpoint's FP8 quantization, which is the folder the
# simulator reads. (The earlier vLLM 0.20.2 profile is kept on branch legacy/vllm-0.20.2.)
#
#   TARGET=r1      ./profiler/profile-deepseek-h200.sh     # DeepSeek-R1        (default)
#   TARGET=v4flash ./profiler/profile-deepseek-h200.sh     # DeepSeek-V4-Flash  (needs DeepGEMM; untested, see feat/deepseek_v4)
#
# R1 staging (recommended for a bounded H200 window):
#   STAGE=coverage ./profiler/profile-deepseek-h200.sh  # 0) check the catalog binds every kernel on this vLLM (no CSVs)
#   SMOKE=1 ./profiler/profile-deepseek-h200.sh      # fast pipeline check (tiny grid, tp=1, plus EP8 rank 0's MoE); verify BEFORE the full run
#   STAGE=fast ./profiler/profile-deepseek-h200.sh   # 1) every category at tp=1 and tp=8, MoE at TP1/EP1 (skew skipped) + meta.yaml
#   STAGE=moe  ./profiler/profile-deepseek-h200.sh   # 2) the TP8/EP8 MoE components, one process per GPU
#   STAGE=skew ./profiler/profile-deepseek-h200.sh   # 3) add the skew sweep (resume-safe; the long pole)
#   STAGE=full ./profiler/profile-deepseek-h200.sh   # fast, moe, then skew (default)
#
# WHY STAGE matters: skew runs per TP (so for --tp 1,8 it runs TWICE) and is the
# long pole. The profiler checkpoints completed shots and resumes an interrupted
# run. STAGE=fast reaches a profile the TP1 deployments (TP1/PP8) can simulate;
# TP8/EP8 deployments also need STAGE=moe. STAGE=skew then adds accuracy and can
# be re-run to resume where it left off.
#
# MoE: R1's routed experts are FP8 with block scales, beside a shared expert.
# The main profile run keeps --ep 1, so its MoE table is TP1/EP1, measured on
# the live native path with the engine's own kernel. A TP8/EP8 deployment runs a
# different kernel -- on Hopper vLLM 0.31 picks Triton for block FP8 at EP=1 but
# FlashInfer CUTLASS at EP>1 (fused_moe/oracle/fp8.py) -- so STAGE=moe measures
# each EP8 rank on the modular path. That path selects the kernel for the target
# and times the shared expert at its TP8 width (docs: Native MoE components ->
# FP8 experts / Shared experts). It runs rank r on GPU r, eight processes at
# once (--moe-ranks), each writing only its own table; the process that
# publishes the last rank logs a rank cross-check. MOE_PARALLEL=0 runs every
# rank in one process on one GPU instead. Per-rank logs go to MOE_LOG_DIR
# (default outputs/profile-deepseek-h200/).
#
# =============================================================================
# WHY THESE PARAMS (not the profiler defaults)
# =============================================================================
# DeepSeek-R1/V4 are long-context MLA + fine-grained-MoE models, much larger
# than the Qwen3/Llama configs this repo shipped with. The knobs were chosen
# against how the simulator LOOKS UP attention (trace_generator): all four axes
# -- prefill_chunk, prefill_key, n_decode, kv_decode -- are blended linearly
# between their two neighbouring grid points.
#
#   Knob                    | default | here (R1) | why
#   ------------------------|---------|-----------|-------------------------------
#   ATTENTION_MAX_KV        | model   | 32768     | covers the measured SWE-bench
#                           | context |           | envelope (max ~21.7K ctx) with
#                           | (164K)  |           | headroom. Below the cap the sim
#                           |         |           | measures; above it it extrapolates.
#                           |         |           | Bump to 65536 if R1 --use-vllm /
#                           |         |           | long-output traces grow past it.
#                           |         |           | (Also widens skew's kvs/kp axes.)
#   ATTENTION_CHUNK_FACTOR  | 2.0     | 2.0 (keep)| the profiler default; the chunk
#                           |         |           | axis is interpolated like the
#                           |         |           | others.
#   MAX_NUM_BATCHED_TOKENS  | 2048    | 4096      | bounds the prefill-chunk axis.
#                           |         |           | SWE-bench's growing prefix +
#                           |         |           | prefix cache keep per-turn new
#                           |         |           | prefill small (~3.8K), so 4096
#                           |         |           | covers it. Match the sim's
#                           |         |           | --max-num-batched-tokens.
#   ATTENTION_KV_FACTOR     | 2.0     | 2.0 (keep)| attention latency is ~linear in
#                           |         |           | KV, so doubling-spaced points
#                           |         |           | interpolate well. KV sits on 2
#                           |         |           | axes, so lowering this is
#                           |         |           | ~quadratic profiling cost for
#                           |         |           | little gain.
#   MAX_NUM_SEQS            | 256     | 256 (keep)| decode-batch (n_decode) axis;
#                           |         |           | raise only to simulate >256
#                           |         |           | concurrent seqs.
#   MEASUREMENT_ITERATIONS  | 3       | 3 (keep)  | N=3 already cuts DVFS jitter to
#                           |         |           | ~5%; more is linear cost for
#                           |         |           | marginal gain on the big shots.
#
# NOTE on skew: skew's kvs/kp axes derive from ATTENTION_MAX_KV, so raising it
# (e.g. to 65536) also grows the skew sweep. If the skew stage runs long, coarsen
# it with SKEW_KVS_FACTOR=4 / SKEW_KP_FACTOR=4 (halves those axes) rather than
# dropping ATTENTION_MAX_KV.
#
# V4-Flash is the OPPOSITE on the KV axis: sliding_window=128 (+ index_topk=512)
# bounds effective attention KV, so a big ATTENTION_MAX_KV is wasted and the skew
# short-circuit fires (t_mean==t_max) -> skew is near-useless. V4 therefore uses
# a SMALL ATTENTION_MAX_KV and --skip-skew, and profiles the sparse indexer in
# two 1-layer passes over compress_ratios (128 then 4).
# =============================================================================
set -euo pipefail

TARGET="${TARGET:-r1}"          # r1 | v4flash
STAGE="${STAGE:-full}"          # full | fast | moe | skew | coverage   (R1 only; V4 is always skip-skew)
HARDWARE="${HARDWARE:-H200}"
MOE_PARALLEL="${MOE_PARALLEL:-1}"   # STAGE=moe: 1 = one process per GPU, 0 = every rank in one process
# Output root (--out-root). Unset writes the profiler default, profiler/perf/.
# SMOKE=1 defaults it to a scratch root: its tiny grid has other acquisition
# settings, which a real run on the same root would refuse to resume.
OUT_ROOT="${OUT_ROOT:-}"
# Leave DTYPE unset: R1 resolves to bf16 activations from its checkpoint, and an
# explicit engine dtype adds an activation suffix to the variant folder (e.g.
# fp8-actbf16) that the simulator never reads.
DTYPE="${DTYPE:-}"
FORCE="${FORCE:-}"             # FORCE=1 -> wipe + re-profile from scratch (default: resume)
VERBOSITY="${VERBOSITY:-}"     # e.g. VERBOSITY=--verbose  (DEBUG + vLLM stdout)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # script lives in profiler/; repo root is its parent
cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
# run_profile <subcommand> <model> <variant|""> [extra flags...]
# Reads the ATTENTION_* / MAX_* / MEASUREMENT_ITERATIONS globals set per target.
# Model-config overrides go in the extra flags as --hf-override KEY=VALUE.
# -----------------------------------------------------------------------------
run_profile() {
  local subcommand="$1"; shift
  local model="$1"; shift
  local variant="$1"; shift
  local cmd=(python3 -m profiler "$subcommand" "$model"
             --hardware "$HARDWARE"
             --tp "$TP_DEGREES"
             --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
             --max-num-seqs "$MAX_NUM_SEQS"
             --attention-max-kv "$ATTENTION_MAX_KV"
             --attention-chunk-factor "$ATTENTION_CHUNK_FACTOR"
             --attention-kv-factor "$ATTENTION_KV_FACTOR"
             --measurement-iterations "$MEASUREMENT_ITERATIONS")
  [[ -n "$DTYPE" ]]          && cmd+=(--dtype "$DTYPE")
  [[ -n "$variant" ]]        && cmd+=(--variant "$variant")
  [[ -n "$OUT_ROOT" ]]       && cmd+=(--out-root "$OUT_ROOT")
  [[ -n "${FORCE:-}" ]]      && cmd+=(--force)
  [[ -n "${VERBOSITY:-}" ]]  && cmd+=($VERBOSITY)
  cmd+=("$@")
  echo "+ ${cmd[*]}"
  "${cmd[@]}"
}

# -----------------------------------------------------------------------------
# run_moe_stage [extra flags...]
# The TP8/EP8 MoE components: `slice --group moe` for EP ranks 0-7. With
# MOE_PARALLEL=1 (default) rank r runs on the r-th visible GPU, all eight at
# once. Each process measures only its own rank (--moe-ranks) and writes only
# that rank's table, so they share one output root; the one that publishes
# the last rank logs the rank cross-check. MOE_PARALLEL=0 measures every rank
# in one process. Returns non-zero if any rank fails.
# -----------------------------------------------------------------------------
run_moe_stage() {
  local ep=8
  local slice=(--tp-refresh "$ep" --group moe --dp 1 --ep "$ep")
  # The slice refreshes tp=8, and the CLI wants tp=1 in the --tp list too.
  # run_profile reads TP_DEGREES; this local shadows the global for it.
  local TP_DEGREES="1,$ep"
  if [[ "$MOE_PARALLEL" != "1" ]]; then
    run_profile slice "$MODEL" "" "${slice[@]}" "$@"
    return
  fi
  local gpus=()
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=, read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"
  else
    mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader)
  fi
  if (( ${#gpus[@]} < ep )); then
    echo "ERROR: STAGE=moe runs one EP rank per GPU and needs $ep GPUs, found ${#gpus[@]}." \
         "Set MOE_PARALLEL=0 to measure every rank in one process." >&2
    return 1
  fi
  local log_dir="${MOE_LOG_DIR:-outputs/profile-deepseek-h200}"
  mkdir -p "$log_dir"
  local pids=() failed=() rank
  for (( rank = 0; rank < ep; rank++ )); do
    echo ">>> EP rank $rank on GPU ${gpus[$rank]}: $log_dir/moe-rank$rank.log"
    ( export CUDA_VISIBLE_DEVICES="${gpus[$rank]}"
      run_profile slice "$MODEL" "" "${slice[@]}" --moe-ranks "$rank" "$@"
    ) > "$log_dir/moe-rank$rank.log" 2>&1 &
    pids+=("$!")
  done
  for (( rank = 0; rank < ep; rank++ )); do
    wait "${pids[$rank]}" || failed+=("$rank")
  done
  if (( ${#failed[@]} )); then
    echo "ERROR: EP ranks ${failed[*]} failed; see $log_dir/moe-rank<rank>.log" >&2
    return 1
  fi
  echo ">>> All $ep EP ranks measured; the last log to finish ends with the rank cross-check."
}

case "$TARGET" in
  # ---------------------------------------------------------------------------
  r1)
    MODEL="deepseek-ai/DeepSeek-R1"
    # tp=1 is MANDATORY (the CLI rejects a --tp list without it): it supplies
    # every tp_stable layer, replicated into tp8/, and the TP1/EP1 MoE table.
    # tp=8 is the deployment sharding; STAGE=moe adds its TP8/EP8 MoE tables.
    # No layer-count override is needed: the profiler reads
    # first_k_dense_replace and boots the 4 layers (3 dense + 1 MoE) that cover
    # both MLP kinds.

    # ---- params (all overridable via env; SMOKE=1 = fast pipeline check) ----
    if [[ "${SMOKE:-}" == "1" ]]; then
      # Tiny 4D grid, tp=1, no skew, 1 iter, then EP8 rank 0's MoE components
      # on a coarse grid -> a few minutes. Verifies the whole path (CSVs
      # written, MLA layers bound, the FP8 expert check and the shared-expert
      # copy at TP8 width) -- NOT real timing numbers, so it writes a scratch root.
      TP_DEGREES="${TP_DEGREES:-1}"; STAGE="fast"
      OUT_ROOT="${OUT_ROOT:-outputs/profile-deepseek-h200/smoke}"
      ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-2048}"
      ATTENTION_CHUNK_FACTOR="${ATTENTION_CHUNK_FACTOR:-2.0}"
      ATTENTION_KV_FACTOR="${ATTENTION_KV_FACTOR:-2.0}"
      MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
      MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-1}"
    else
      TP_DEGREES="${TP_DEGREES:-1,8}"
      # Sized to the SWE-bench trace (max ~21.7K ctx). For a bounded window,
      # MAX_NUM_SEQS=64 (SWE-bench is low-concurrency) shrinks the attention
      # grid; the run's heartbeat logs the shot count and ETA. Raise
      # ATTENTION_MAX_KV to 65536 only for longer traces (measure first).
      ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-32768}"
      ATTENTION_CHUNK_FACTOR="${ATTENTION_CHUNK_FACTOR:-2.0}"
      ATTENTION_KV_FACTOR="${ATTENTION_KV_FACTOR:-2.0}"
      MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
      MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-3}"
    fi
    # skew density (only consulted when skew runs). Bump kvs/kp to 4.0 if the
    # skew sweep is too slow at ATTENTION_MAX_KV=65536.
    SKEW_N_FACTOR="${SKEW_N_FACTOR:-2.0}"
    SKEW_PC_FACTOR="${SKEW_PC_FACTOR:-2.0}"
    SKEW_KP_FACTOR="${SKEW_KP_FACTOR:-2.0}"
    SKEW_KVS_FACTOR="${SKEW_KVS_FACTOR:-2.0}"

    skew_factor_flags=(--skew-n-factor "$SKEW_N_FACTOR"
                       --skew-pc-factor "$SKEW_PC_FACTOR"
                       --skew-kp-factor "$SKEW_KP_FACTOR"
                       --skew-kvs-factor "$SKEW_KVS_FACTOR")

    case "$STAGE" in
      coverage)
        # Structural check at tp=1: does the catalog bind every kernel this vLLM
        # launches? Writes no CSVs. Run it first after any vLLM upgrade. It exits
        # non-zero while any gap remains, including the model runner's own
        # "(unscoped)" input-preparation kernels, which no catalog entry binds:
        # read the listed gaps and over-matches, not just the exit status.
        echo ">>> Catalog coverage for DeepSeek-R1 on $HARDWARE"
        python3 -m profiler coverage "$MODEL" --hardware "$HARDWARE"
        exit 0 ;;
      fast|moe|skew|full) ;;
      *) echo "ERROR: unknown STAGE='$STAGE' (use full|fast|moe|skew|coverage)" >&2; exit 2 ;;
    esac

    echo ">>> Profiling DeepSeek-R1 on $HARDWARE  (tp=$TP_DEGREES, STAGE=$STAGE)"
    # --ep 1 keeps the main run's MoE table at TP1/EP1; STAGE=moe adds TP8/EP8.
    if [[ "$STAGE" == fast || "$STAGE" == full ]]; then
      run_profile profile "$MODEL" "" --ep 1 --skip-skew
    fi
    if [[ "${SMOKE:-}" == "1" ]]; then
      echo ">>> Smoke: EP8 rank 0's MoE components (coarse grid, one round)"
      TP_DEGREES="1,8" run_profile slice "$MODEL" "" --tp-refresh 8 --group moe \
        --dp 1 --ep 8 --moe-ranks 0 --moe-rounds 1 --moe-grid-factor 8
    elif [[ "$STAGE" == moe || "$STAGE" == full ]]; then
      run_moe_stage
    fi
    # No --ep here: the CLI refuses native MoE options with --only-skew.
    if [[ "$STAGE" == skew || "$STAGE" == full ]]; then
      run_profile profile "$MODEL" "" --only-skew "${skew_factor_flags[@]}"
    fi
    out="${OUT_ROOT:-profiler/perf}/${HARDWARE}/${MODEL}/fp8"
    echo ">>> Done. Output: ${out}/tp{1,8}/"
    echo ">>> Smoke-check the MLA layers were captured:"
    echo "    cut -d, -f1 ${out}/tp1/dense.csv | sort -u | grep -E 'mla_qkv_a_proj|mla_a_layernorm|mla_q_b_proj|mla_glue'"
    ;;

  # ---------------------------------------------------------------------------
  v4flash)
    MODEL="deepseek-ai/DeepSeek-V4-Flash"
    TP_DEGREES="1,8"

    # V4's mhc_* hyper-connection ops call deep_gemm.* directly, so even a
    # default forward pass needs DeepGEMM. Fail fast with a clear hint.
    if ! python3 -c 'import deep_gemm' >/dev/null 2>&1; then
      echo "ERROR: DeepGEMM not importable. Install it first:" >&2
      echo "       bash scripts/install-deepgemm.sh" >&2
      exit 1
    fi

    # ---- decided params (rationale in the header) ----
    # sliding_window=128 (+ index_topk=512) bounds effective KV -> keep KV small
    # and skip skew (it short-circuits to mean==max here).
    ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-2048}"
    ATTENTION_CHUNK_FACTOR="${ATTENTION_CHUNK_FACTOR:-1.5}"
    ATTENTION_KV_FACTOR="${ATTENTION_KV_FACTOR:-2.0}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
    MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-3}"

    # V4 is all-MoE, so a 2-layer pass would materialise TWO FusedMoE and the
    # hook raises "got 2" -> use 1-layer passes. Two passes over the indexer's
    # compress_ratios: 128 (non-indexer/dense-ish) then 4 (DeepseekV4Indexer +
    # sparse MLA). num_key_value_heads:8 so tp=8's shard check (1 % 8 != 0)
    # doesn't crash — MLA replicates its latent KV rather than sharding it, so
    # per-rank KV stays 1 and the timing is unaffected.
    echo ">>> Profiling DeepSeek-V4-Flash on $HARDWARE  (tp=$TP_DEGREES, two-pass indexer, skew skipped)"
    run_profile profile "$MODEL" "bf16-m128" --hf-override num_hidden_layers=1 \
      --hf-override 'compress_ratios=[128]' --hf-override num_key_value_heads=8 --skip-skew
    run_profile profile "$MODEL" "bf16-m4" --hf-override num_hidden_layers=1 \
      --hf-override 'compress_ratios=[4]' --hf-override num_key_value_heads=8 --skip-skew
    echo ">>> Done. Output: profiler/perf/${HARDWARE}/${MODEL}/{bf16-m128,bf16-m4}/tp{1,8}/"
    ;;

  *)
    echo "ERROR: unknown TARGET='$TARGET' (use r1|v4flash)" >&2
    exit 2
    ;;
esac
