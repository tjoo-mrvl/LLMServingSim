#!/usr/bin/env bash
# =============================================================================
# profile-deepseek-h200.sh — DeepSeek-R1 layerwise profiler driver
# -----------------------------------------------------------------------------
# Run INSIDE the vLLM/profiler container (scripts/docker-vllm.sh, vLLM 0.31),
# from the repo root. Each profiler process uses ONE GPU + dummy weights
# (load_format=dummy); STAGE=moe runs one such process per GPU. It reads
# the vendored configs/model/deepseek-ai/*.json and downloads NO
# weights. R1 output lands in profiler/perf/H200/deepseek-ai/DeepSeek-R1/fp8/:
# the variant follows the checkpoint's FP8 quantization, which is the folder the
# simulator reads. (The earlier vLLM 0.20.2 profile is kept on branch legacy/vllm-0.20.2.)
#
# R1 staging (recommended for a bounded H200 window):
#   STAGE=coverage ./profiler/profile-deepseek-h200.sh  # 0) check the catalog binds every kernel on this vLLM (no CSVs)
#   SMOKE=1 ./profiler/profile-deepseek-h200.sh      # fast pipeline check (tiny grid, tp=1, plus EP8 rank 0's MoE); verify BEFORE the full run
#   STAGE=fast ./profiler/profile-deepseek-h200.sh   # 1) every category at tp=1 and tp=8, MoE at TP1/EP1 (skew skipped) + meta.yaml
#   STAGE=moe  ./profiler/profile-deepseek-h200.sh   # 2) the TP8/EP8 MoE components (profile-moe-ranks.sh)
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
# FP8 experts / Shared experts). It runs profiler/profile-moe-ranks.sh with
# TP=EP=${EP:-8}, DP=1: one process per visible GPU (--moe-ranks), each writing
# only its own ranks' tables; the process that publishes the last rank logs a
# rank cross-check. With fewer GPUs than ranks each GPU measures several in
# turn, so CUDA_VISIBLE_DEVICES=0 measures all eight on GPU 0. Per-process logs
# go to MOE_LOG_DIR (default outputs/profile-deepseek-h200/).
#
# =============================================================================
# WHY THESE PARAMS (not the profiler defaults)
# =============================================================================
# DeepSeek-R1 is a long-context MLA + fine-grained-MoE model, much larger
# than the Qwen3/Llama configs this repo shipped with. The knobs were chosen
# against how the simulator LOOKS UP attention (trace_generator): all four axes
# -- prefill_chunk, prefill_key, n_decode, kv_decode -- are blended linearly
# between their two neighbouring grid points.
#
#   Knob                    | default | here (R1) | why
#   ------------------------|---------|-----------|-------------------------------
#   ATTENTION_MAX_KV        | model   | unset     | the profiler's default: R1's whole
#                           | context |           | context window (~164K), as every
#                           | (164K)  |           | shipped long-context bundle uses.
#                           |         |           | Below the reach the sim measures;
#                           |         |           | above it it extrapolates, and
#                           |         |           | nothing warns. Set it only to
#                           |         |           | bound a trial run.
#   ATTENTION_CHUNK_FACTOR  | 2.0     | 2.0 (keep)| the profiler default; the chunk
#                           |         |           | axis is interpolated like the
#                           |         |           | others.
#   MAX_NUM_BATCHED_TOKENS  | 2048    | 16384     | bounds the prefill-chunk axis and
#                           |         |           | the MoE token axis. 16384 is vLLM
#                           |         |           | 0.31's offline default on an H200
#                           |         |           | and holds a ~10K accessibility-tree
#                           |         |           | prompt (OSWorld) in one step. Profile
#                           |         |           | at least as high as the bench and
#                           |         |           | the sim use (--max-num-batched-tokens).
#   ATTENTION_KV_FACTOR     | 2.0     | 2.0 (keep)| attention latency is ~linear in
#                           |         |           | KV, so doubling-spaced points
#                           |         |           | interpolate well. KV sits on 2
#                           |         |           | axes, so lowering this is
#                           |         |           | ~quadratic profiling cost for
#                           |         |           | little gain.
#   MAX_NUM_SEQS            | 256     | 512       | decode-batch (n_decode) axis.
#                           |         |           | Profile ABOVE the runtime cap: a
#                           |         |           | mixed (prefill + decode) shot at
#                           |         |           | n = MSQ needs MSQ+1 requests and is
#                           |         |           | filtered, so a profile at 256 has no
#                           |         |           | mixed shot above 181 decodes. Run the
#                           |         |           | bench and sim at <=256 (upstream's
#                           |         |           | examples profile 256, run 128).
#   MEASUREMENT_ITERATIONS  | 3       | 3 (keep)  | N=3 already cuts DVFS jitter to
#                           |         |           | ~5%; more is linear cost for
#                           |         |           | marginal gain on the big shots.
#
# =============================================================================
# STAGED RUN: a small trial first, then extend it
# =============================================================================
# Resume extends the ordinary tables when the limits grow, so a trial pass
# costs nothing the full pass redoes. Checked on an A10G (Llama-3.1-8B and
# Qwen3-30B at tiny nested limits) and by enumerating R1's TP8 grids:
#   * dense, per_sequence, attention: shots already in the CSV are skipped.
#     R1's 2048/128 plan has 22,312 attention shots at the full context reach,
#     all inside the 16384/512 plan of 40,022 (14,088 with ATTENTION_MAX_KV=
#     16384, of which 92% are inside it). Rows outside the final plan stay.
#   * skew: only part of a smaller sweep is reused (76% of R1's 2048/128 cases,
#     31% in the toy), so run STAGE=skew once, at the final limits.
#   * native MoE: a different limit is a different plan, and resume refuses it
#     ("use a new output root"). Run STAGE=moe once, at the final limits, in the
#     real root. A trial that ran STAGE=fast leaves its TP1/EP1 MoE table at the
#     trial's limits, so the extending STAGE=fast ends with that refusal after the
#     ordinary tables, TP-stable copies and meta.yaml are complete.
# Never set FORCE for the extension: it replaces what was measured.
#
#   MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=128 ATTENTION_MAX_KV=16384 \
#       STAGE=fast ./profiler/profile-deepseek-h200.sh      # trial
#   STAGE=fast ./profiler/profile-deepseek-h200.sh          # extend to 16384/512
#   STAGE=moe  ./profiler/profile-deepseek-h200.sh          # once, final limits
#   STAGE=skew ./profiler/profile-deepseek-h200.sh          # once, final limits
#
# NOTE on skew: skew's kvs/kp axes derive from ATTENTION_MAX_KV, so the full
# context also widens the skew sweep. If the skew stage runs long, coarsen it
# with SKEW_KVS_FACTOR=4 / SKEW_KP_FACTOR=4 (halves those axes) rather than
# lowering ATTENTION_MAX_KV, which would make the attention table extrapolate.
# =============================================================================
set -euo pipefail

TARGET="${TARGET:-r1}"          # r1
STAGE="${STAGE:-full}"          # full | fast | moe | skew | coverage
HARDWARE="${HARDWARE:-H200}"
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
# set_run_flags
# Fills run_flags with the flags every command here shares: the ATTENTION_* /
# MAX_* / MEASUREMENT_ITERATIONS globals set per target, plus DTYPE, FORCE and
# VERBOSITY. --hardware, --tp and --out-root are each caller's own.
# -----------------------------------------------------------------------------
set_run_flags() {
  run_flags=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
             --max-num-seqs "$MAX_NUM_SEQS"
             --attention-chunk-factor "$ATTENTION_CHUNK_FACTOR"
             --attention-kv-factor "$ATTENTION_KV_FACTOR"
             --measurement-iterations "$MEASUREMENT_ITERATIONS")
  [[ -n "${ATTENTION_MAX_KV:-}" ]] && run_flags+=(--attention-max-kv "$ATTENTION_MAX_KV")
  [[ -n "$DTYPE" ]]          && run_flags+=(--dtype "$DTYPE")
  [[ -n "${FORCE:-}" ]]      && run_flags+=(--force)
  [[ -n "${VERBOSITY:-}" ]]  && run_flags+=($VERBOSITY)
  return 0
}

# -----------------------------------------------------------------------------
# run_profile <subcommand> <model> [extra flags...]
# Model-config overrides go in the extra flags as --hf-override KEY=VALUE.
# -----------------------------------------------------------------------------
run_profile() {
  local subcommand="$1"; shift
  local model="$1"; shift
  local run_flags
  set_run_flags
  local cmd=(python3 -m profiler "$subcommand" "$model"
             --hardware "$HARDWARE"
             --tp "$TP_DEGREES"
             "${run_flags[@]}")
  [[ -n "$OUT_ROOT" ]]       && cmd+=(--out-root "$OUT_ROOT")
  cmd+=("$@")
  echo "+ ${cmd[*]}"
  "${cmd[@]}"
}

# -----------------------------------------------------------------------------
# run_moe_stage [extra flags...]
# The TP8/EP8 MoE components, through profile-moe-ranks.sh: one process per
# visible GPU, each measuring only its own EP ranks (--moe-ranks) on one shared
# output root. Returns non-zero if any process fails.
# -----------------------------------------------------------------------------
run_moe_stage() {
  local ep="${EP:-8}"
  local run_flags
  set_run_flags
  MODEL="$MODEL" HARDWARE="$HARDWARE" TP="$ep" DP=1 EP="$ep" OUT_ROOT="$OUT_ROOT" \
    LOG_DIR="${MOE_LOG_DIR:-outputs/profile-deepseek-h200}" \
    "$REPO_ROOT/profiler/profile-moe-ranks.sh" "${run_flags[@]}" "$@"
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
      # ATTENTION_MAX_KV stays unset, so the attention and skew grids reach
      # R1's whole context window (the profiler default). For a bounded
      # window, MAX_NUM_SEQS=64 (SWE-bench is low-concurrency) shrinks the
      # attention grid; the run's heartbeat logs the shot count and ETA.
      ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-}"
      ATTENTION_CHUNK_FACTOR="${ATTENTION_CHUNK_FACTOR:-2.0}"
      ATTENTION_KV_FACTOR="${ATTENTION_KV_FACTOR:-2.0}"
      MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
      MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-3}"
    fi
    # skew density (only consulted when skew runs). Bump kvs/kp to 4.0 if the
    # skew sweep is too slow at the full context.
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
      run_profile profile "$MODEL" --ep 1 --skip-skew
    fi
    if [[ "${SMOKE:-}" == "1" ]]; then
      echo ">>> Smoke: EP8 rank 0's MoE components (coarse grid, one round)"
      TP_DEGREES="1,8" run_profile slice "$MODEL" --tp-refresh 8 --group moe \
        --dp 1 --ep 8 --moe-ranks 0 --moe-rounds 1 --moe-grid-factor 8
    elif [[ "$STAGE" == moe || "$STAGE" == full ]]; then
      run_moe_stage
    fi
    # No --ep here: the CLI refuses native MoE options with --only-skew.
    if [[ "$STAGE" == skew || "$STAGE" == full ]]; then
      run_profile profile "$MODEL" --only-skew "${skew_factor_flags[@]}"
    fi
    out="${OUT_ROOT:-profiler/perf}/${HARDWARE}/${MODEL}/fp8"
    echo ">>> Done. Output: ${out}/tp{1,8}/"
    echo ">>> Smoke-check the MLA layers were captured:"
    echo "    cut -d, -f1 ${out}/tp1/dense.csv | sort -u | grep -E 'mla_qkv_a_proj|mla_a_layernorm|mla_q_b_proj|mla_glue'"
    ;;

  *)
    echo "ERROR: unknown TARGET='$TARGET' (use r1)" >&2
    exit 2
    ;;
esac
