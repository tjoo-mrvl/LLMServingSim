#!/usr/bin/env bash
# =============================================================================
# profile-deepseek-h200.sh — DeepSeek-R1 / V4-Flash layerwise profiler driver
# -----------------------------------------------------------------------------
# Run INSIDE the vLLM/profiler container (scripts/docker-vllm.sh, vLLM 0.31),
# from the repo root. Profiling uses ONE GPU + dummy weights (load_format=dummy)
# — it reads the vendored configs/model/deepseek-ai/*.json and downloads NO
# weights. R1 output lands in profiler/perf/H200/deepseek-ai/DeepSeek-R1/fp8/:
# the variant follows the checkpoint's FP8 quantization, which is the folder the
# simulator reads. (The earlier vLLM 0.20.2 profile is kept on branch legacy/vllm-0.20.2.)
#
#   TARGET=r1      ./profiler/profile-deepseek-h200.sh     # DeepSeek-R1        (default)
#   TARGET=v4flash ./profiler/profile-deepseek-h200.sh     # DeepSeek-V4-Flash  (needs DeepGEMM; untested, see feat/deepseek_v4)
#
# R1 staging (recommended for a bounded H200 window):
#   STAGE=coverage ./profiler/profile-deepseek-h200.sh  # 0) check the catalog binds every kernel on this vLLM (no CSVs)
#   SMOKE=1 ./profiler/profile-deepseek-h200.sh      # fast tp=1 pipeline check (tiny grid); verify BEFORE the full run
#   STAGE=fast ./profiler/profile-deepseek-h200.sh   # 1) complete, SIMULATABLE profile (skew skipped) + meta.yaml
#   STAGE=skew ./profiler/profile-deepseek-h200.sh   # 2) add the skew sweep (resume-safe; the long pole)
#   STAGE=full ./profiler/profile-deepseek-h200.sh   # everything in one shot (default)
#
# WHY STAGE matters: skew runs per TP (so for --tp 1,8 it runs TWICE) and is the
# long pole. The profiler checkpoints completed shots and resumes an interrupted
# run, but STAGE=fast still reaches a usable, simulatable profile first; STAGE=skew
# then adds accuracy and can be re-run to resume where it left off.
#
# MoE: R1's MoE is quantized and has a shared expert, which the profiler's live
# native components measure at TP1/DP1/EP1 only, so the run is pinned to --ep 1
# and the simulator prices EP>1 ranks from that table. On Hopper, vLLM 0.31's
# automatic FP8 MoE backend is Triton at EP=1 but FlashInfer CUTLASS at EP>1
# (fused_moe/oracle/fp8.py), and the profiler cannot pin moe_backend, so this
# profile times Triton while a TP8/EP8 bench runs FlashInfer CUTLASS. Measure the
# gap against an EP=8 bench before trusting MoE-bound results. (vLLM 0.31 has no
# VLLM_USE_FLASHINFER_MOE_FP8; the backend is KernelConfig.moe_backend.)
#
# =============================================================================
# WHY THESE PARAMS (not the profiler defaults)
# =============================================================================
# DeepSeek-R1/V4 are long-context MLA + fine-grained-MoE models, much larger
# than the Qwen3/Llama configs this repo shipped with. The knobs were chosen
# against how the simulator actually LOOKS UP attention (trace_generator):
#   * prefill_chunk, n_decode : NEAREST-NEIGHBOUR (no interpolation)
#   * kv_prefill, kv_decode   : BILINEAR (interpolates between grid points)
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
#   ATTENTION_CHUNK_FACTOR  | 2.0     | 1.5       | chunk is NEAREST-NEIGHBOUR and
#                           |         |           | prefill cost is ~quadratic in
#                           |         |           | chunk, so coarse spacing snaps
#                           |         |           | a runtime chunk to a wrong cost.
#                           |         |           | Densify. (chunk is 1 axis ->
#                           |         |           | cheap to densify.)
#   MAX_NUM_BATCHED_TOKENS  | 2048    | 4096      | bounds the (nearest-neighbour)
#                           |         |           | prefill-chunk axis. SWE-bench's
#                           |         |           | growing prefix + prefix cache
#                           |         |           | keep per-turn new prefill small
#                           |         |           | (~3.8K), so 4096 covers it. Match
#                           |         |           | the sim's --max-num-batched-tokens.
#   ATTENTION_KV_FACTOR     | 2.0     | 2.0 (keep)| KV axes are BILINEAR and attn
#                           |         |           | latency is ~linear in KV, so
#                           |         |           | doubling-spaced points interp
#                           |         |           | well. KV sits on 2 axes, so
#                           |         |           | lowering this is ~quadratic
#                           |         |           | profiling cost for little gain.
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
STAGE="${STAGE:-full}"          # full | fast | skew | coverage   (R1 only; V4 is always skip-skew)
HARDWARE="${HARDWARE:-H200}"
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
  [[ -n "${FORCE:-}" ]]      && cmd+=(--force)
  [[ -n "${VERBOSITY:-}" ]]  && cmd+=($VERBOSITY)
  cmd+=("$@")
  echo "+ ${cmd[*]}"
  "${cmd[@]}"
}

case "$TARGET" in
  # ---------------------------------------------------------------------------
  r1)
    MODEL="deepseek-ai/DeepSeek-R1"
    # tp=1 is MANDATORY (the CLI rejects a --tp list without it): it supplies
    # the MoE profile and every tp_stable layer, which get replicated into tp8/.
    # tp=8 is the deployment sharding. No layer-count override is needed: the
    # profiler reads first_k_dense_replace and boots the 4 layers (3 dense + 1
    # MoE) that cover both MLP kinds.

    # ---- params (all overridable via env; SMOKE=1 = fast pipeline check) ----
    if [[ "${SMOKE:-}" == "1" ]]; then
      # Tiny 4D grid, tp=1, no skew, 1 iter -> a few minutes. Verifies the whole
      # path (CSVs written, MLA layers bound) — NOT real timing numbers.
      TP_DEGREES="${TP_DEGREES:-1}"; STAGE="fast"
      ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-2048}"
      ATTENTION_CHUNK_FACTOR="${ATTENTION_CHUNK_FACTOR:-2.0}"
      ATTENTION_KV_FACTOR="${ATTENTION_KV_FACTOR:-2.0}"
      MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
      MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-1}"
    else
      TP_DEGREES="${TP_DEGREES:-1,8}"
      # Sized to the SWE-bench trace (max ~21.7K ctx). NOTE: chunk-factor 1.5 +
      # MSQ 256 makes a ~19k-shot attention grid (~4h/TP). For a bounded window,
      # override ATTENTION_CHUNK_FACTOR=2.0 and MAX_NUM_SEQS=64 (SWE-bench is
      # low-concurrency) to cut it to ~1-1.5h/TP. Raise ATTENTION_MAX_KV to 65536
      # only for longer traces (measure first).
      ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-32768}"
      ATTENTION_CHUNK_FACTOR="${ATTENTION_CHUNK_FACTOR:-1.5}"
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
        # launches? Writes no CSVs. Run it first after any vLLM upgrade.
        echo ">>> Catalog coverage for DeepSeek-R1 on $HARDWARE"
        python3 -m profiler coverage "$MODEL" --hardware "$HARDWARE"
        exit 0 ;;
      fast) stage_flags=(--skip-skew) ;;
      skew) stage_flags=(--only-skew "${skew_factor_flags[@]}") ;;
      full) stage_flags=("${skew_factor_flags[@]}") ;;
      *)    echo "ERROR: unknown STAGE='$STAGE' (use full|fast|skew|coverage)" >&2; exit 2 ;;
    esac

    echo ">>> Profiling DeepSeek-R1 on $HARDWARE  (tp=$TP_DEGREES, STAGE=$STAGE)"
    run_profile profile "$MODEL" "" --ep 1 "${stage_flags[@]}"
    echo ">>> Done. Output: profiler/perf/${HARDWARE}/${MODEL}/fp8/tp{1,8}/"
    echo ">>> Smoke-check the MLA layers were captured:"
    echo "    cut -d, -f1 profiler/perf/${HARDWARE}/${MODEL}/fp8/tp1/dense.csv | sort -u | grep -E 'mla_qkv_a_proj|mla_a_layernorm|mla_b_proj'"
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
