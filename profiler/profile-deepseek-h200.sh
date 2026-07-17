#!/usr/bin/env bash
# =============================================================================
# profile-deepseek-h200.sh — DeepSeek-R1 / V4-Flash layerwise profiler driver
# -----------------------------------------------------------------------------
# Run INSIDE the vLLM/profiler container (PODMAN_SETUP.md §2), from the repo
# root. Profiling uses ONE GPU + dummy weights (load_format=dummy) — it reads
# the vendored configs/model/deepseek-ai/*.json and downloads NO weights.
#
#   TARGET=r1      ./profiler/profile-deepseek-h200.sh     # DeepSeek-R1        (default)
#   TARGET=v4flash ./profiler/profile-deepseek-h200.sh     # DeepSeek-V4-Flash  (needs DeepGEMM)
#
# R1 staging (recommended for a bounded H200 window):
#   STAGE=fast ./profiler/profile-deepseek-h200.sh   # 1) complete, SIMULATABLE profile (skew skipped) + meta.yaml
#   STAGE=skew ./profiler/profile-deepseek-h200.sh   # 2) add the skew sweep (resume-safe; the long pole)
#   STAGE=full ./profiler/profile-deepseek-h200.sh   # everything in one shot (default)
#
# WHY STAGE matters (from runner.py): meta.yaml is written and tp_stable layers
# are replicated into tp8/ only AFTER every TP finishes, and skew runs per-TP
# (so for --tp 1,8 it runs TWICE). A STAGE=full run interrupted before it
# completes leaves no meta.yaml (the simulator can't load the profile) and no
# tp8 data. STAGE=fast reaches a usable, simulatable milestone first; STAGE=skew
# then adds accuracy and can be re-run to resume where it left off.
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
#   ATTENTION_MAX_KV        | 16384   | 32768     | covers the measured SWE-bench
#                           |         |           | envelope (max ~21.7K ctx) with
#                           |         |           | headroom. Below the cap the sim
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
STAGE="${STAGE:-full}"          # full | fast | skew   (R1 only; V4 is always skip-skew)
HARDWARE="${HARDWARE:-H200}"
DTYPE="${DTYPE:-bfloat16}"      # bf16 activations; vLLM auto-applies the fp8 weights from quantization_config
FORCE="${FORCE:-}"             # FORCE=1 -> wipe + re-profile from scratch (default: resume)
VERBOSITY="${VERBOSITY:-}"     # e.g. VERBOSITY=--verbose  (DEBUG + vLLM stdout)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # script lives in profiler/; repo root is its parent
cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
# run_profile <model> <variant|""> <hf_overrides_json|""> [extra flags...]
# Reads the ATTENTION_* / MAX_* / MEASUREMENT_ITERATIONS globals set per target.
# -----------------------------------------------------------------------------
run_profile() {
  local model="$1"; shift
  local variant="$1"; shift
  local hf_overrides="$1"; shift
  local cmd=(python3 -m profiler profile "$model"
             --hardware "$HARDWARE"
             --tp "$TP_DEGREES"
             --dtype "$DTYPE"
             --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
             --max-num-seqs "$MAX_NUM_SEQS"
             --attention-max-kv "$ATTENTION_MAX_KV"
             --attention-chunk-factor "$ATTENTION_CHUNK_FACTOR"
             --attention-kv-factor "$ATTENTION_KV_FACTOR"
             --measurement-iterations "$MEASUREMENT_ITERATIONS")
  [[ -n "$hf_overrides" ]]   && cmd+=(--hf-overrides "$hf_overrides")
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
    # the MoE profile (profiled only at tp=1) and every tp_stable layer, which
    # get replicated into tp8/. tp=8 is the deployment sharding.
    TP_DEGREES="1,8"
    # 2-layer override so ONE run materialises a dense (layer 0) AND an MoE
    # (layer 1) block; default num_hidden_layers=1 + R1's first_k_dense_replace=3
    # would give no MoE layer -> the MoE hook raises "got 0".
    HF_OVERRIDES='{"num_hidden_layers": 2, "first_k_dense_replace": 1}'

    # ---- decided params (rationale in the header) ----
    # Sized to the SWE-bench trace (max ~21.7K ctx). Raise ATTENTION_MAX_KV to
    # 65536 if you regenerate with R1 --use-vllm / longer outputs (measure first).
    ATTENTION_MAX_KV=32768
    ATTENTION_CHUNK_FACTOR=1.5
    ATTENTION_KV_FACTOR=2.0
    MAX_NUM_BATCHED_TOKENS=4096
    MAX_NUM_SEQS=256
    MEASUREMENT_ITERATIONS=3
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
      fast) stage_flags=(--skip-skew) ;;
      skew) stage_flags=(--only-skew "${skew_factor_flags[@]}") ;;
      full) stage_flags=("${skew_factor_flags[@]}") ;;
      *)    echo "ERROR: unknown STAGE='$STAGE' (use full|fast|skew)" >&2; exit 2 ;;
    esac

    echo ">>> Profiling DeepSeek-R1 on $HARDWARE  (tp=$TP_DEGREES, STAGE=$STAGE)"
    run_profile "$MODEL" "" "$HF_OVERRIDES" "${stage_flags[@]}"
    echo ">>> Done. Output: profiler/perf/${HARDWARE}/${MODEL}/bf16/tp{1,8}/"
    echo ">>> Smoke-check the MLA layers were captured (repr_match):"
    echo "    cut -d, -f1 profiler/perf/${HARDWARE}/${MODEL}/bf16/tp1/dense.csv | sort -u | grep -E 'q_a_layernorm|kv_a_layernorm|q_b_proj|kv_b_proj'"
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
    run_profile "$MODEL" "bf16-m128" '{"num_hidden_layers":1,"compress_ratios":[128],"num_key_value_heads":8}' --skip-skew
    run_profile "$MODEL" "bf16-m4"   '{"num_hidden_layers":1,"compress_ratios":[4],"num_key_value_heads":8}'   --skip-skew
    echo ">>> Done. Output: profiler/perf/${HARDWARE}/${MODEL}/{bf16-m128,bf16-m4}/tp{1,8}/"
    ;;

  *)
    echo "ERROR: unknown TARGET='$TARGET' (use r1|v4flash)" >&2
    exit 2
    ;;
esac
