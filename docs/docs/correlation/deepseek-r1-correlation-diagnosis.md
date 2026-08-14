# DeepSeek-R1 decode over-prediction — correlation diagnosis (post-`f1384c7`)

**Date:** 2026-08-11 · **Branch:** `feat/deepseek_v3` · **HEAD:** `dadd3f3` (in sync with origin)
**Scope:** why the simulator *over*-predicts R1 decode TPOT vs real vLLM after the MoE fixes, localized
to a specific term with per-batch attribution. **No simulator code was changed in producing this doc.**

This supersedes/extends the prior investigation in `next_steps_deepseek_r1.md` §11–§15: those runs were
**pre-`f1384c7`** and decomposed **b64 only**. This pass is fresh (post-`f1384c7`), adds **b1 and b16**,
and **corrects an overclaim** (that an analytical fix "closes b1 to ±2%" — it does not; see §7).

---

## 1. TL;DR

- **b1 (+88% TPOT) is definitively the `_lookup_moe` sub-top_k floor** — not another simulator term.
  Proof is methodology-independent: **the sim's MoE term alone (11.80 ms) exceeds bench's *entire* b1
  step (9.82 ms).** It charges `moe(1,8)=203.5µs`/layer × 58 because `route_ep` yields `activated=1`
  per rank and `_lookup_moe` clamps `activated<8` up to the 8-expert row.
- **b16/b64 (+33% / +22%)** look up **covered, genuinely-profiled** points (`moe(2,16)`, `moe(8,32)`) —
  *not* extrapolations. The identified MoE error there is the **activated-expert over-count** (the
  formula assumes perfect load balance), but it is **modest** (~10% of the MoE term, ~1.7 ms at b16,
  ~2.4 ms at b64) and does **not** close the gap. The remainder is **confounded** (see §6) and not yet
  cleanly attributed.
- **MLA-attention is accurate** at every batch (b64: 10.31 ms, matches `next_steps` §15.1.3's physical
  floor analysis). Attention is *not* the problem.
- **The correct low-batch data point `moe(1,1)` is unmeasurable** (the profiler structurally routes
  every token to `top_k=8` experts). Any b1 fix is therefore a **model/estimate**, and un-clamping alone
  likely only moves b1 from +88% to ~+45%, because a second, separate term — the tp1-measured fixed
  gate/permute overhead over-counted for an EP rank — is *not* addressed by un-clamping (§7).

---

## 2. Status after `f1384c7`

Matched-batch fixed **8192-in / 4096-out** sweep, sim vs real vLLM (8×H200, TP8/EP8, median):

| batch | TTFT | TPOT | bench TPOT | sim TPOT |
|-------|------|------|-----------|----------|
| b1  | +6%  | **+88%** | 9.82 ms | 18.48 ms |
| b16 | +14% | **+33%** | 20.1 ms | 26.80 ms |
| b64 | +13% | **+22%** | 39.77 ms | 48.61 ms |
| ShareGPT-R1 | (queue-coupled) | **+21%** | — | — |

**TTFT (prefill) is sound.** The residual is **decode-only**, over-predicting, and *shrinks* as batch
grows (+88% → +33% → +22%).

### What the two committed fixes did
- **`f1384c7`** — corrected the per-rank **tokens-key**: `local_tokens = total_len/ep_size` (was
  `total_len·hit_prob`, i.e. `ep·hit_prob ≈ 5.25×` too large for R1 ep8/k8). This removed the long-context
  prefill blow-up. It did **not** touch the activated-count or the sub-top_k floor.
- **`dadd3f3`** — per-iteration `.et` trace reaping (disk), unrelated to accuracy.

---

## 3. How the `moe(x, y)` lookup works

`moe.csv` is profiled at **tp=1**. A row `moe[T, A]` is the *measured* latency of one MoE layer where
**T tokens each fire `top_k=8` experts, landing on A distinct experts** — i.e. `T·top_k` grouped-GEMM
rows over A experts, plus a fixed gate-over-all-256-experts + permute + kernel-launch overhead.
(`profiler/core/hooks/moe_hook.py::ExpertRoute.forge` builds `(num_tokens, top_k)` routing;
`profiler/core/categories.py:496` skips `activated < top_k`, so the smallest profiled `A` is 8.)

At runtime, `serving/core/gate_function.py::_balanced_route_ep` converts a decode batch of **N** requests
into the **per-EP-rank** operating point `(x = local_tokens, y = activated_experts)` for `ep=8, k=8,
E_rank = 256/8 = 32`:

- `pairs_per_rank = N·k/ep = N`
- `x = local_tokens = max(1, round(pairs_per_rank / k)) = max(1, round(N/8))`
- `y = activated_experts = min(round(pairs_per_rank), E_rank) = min(round(N), 32)`

| batch N | x = local | y = activated | lookup | physical meaning |
|---|---|---|---|---|
| **b1**  | 1/8 → **1** (floored) | 1 → **8** (clamped) | `moe(1,8)` = 203.5 µs | *should* be 1 token → 1 expert (`moe(1,1)`) |
| **b16** | **2** | **16** | `moe(2,16)` = 286.7 µs | 2 tokens' work over 16 experts |
| **b64** | **8** | **32** | `moe(8,32)` = 458.0 µs | 8 tokens' work over 32 experts |

Per MoE layer this latency is charged once; R1 has **58 MoE layers** (`num_hidden_layers=61`,
`first_k_dense_replace=3`). The 8 EP ranks run in parallel (ALLGATHER/REDUCESCATTER), so the per-step
MoE critical path = `58 × moe(x,y)`.

**The b1 clamp is the crux.** Physically, at ep8 a single decode token's `top_k=8` experts scatter
across up to 8 ranks (≈1 each), so a rank truly does "1 token → **1** expert" = `moe(1,1)`. But the
profiler cannot represent `activated<8`, and `_lookup_moe` (`trace_generator.py:844–847`, via the
`bisect`→index-0 branch of `_lookup_bounds:471–472`) **hard-clamps** the request up to `moe(1,8)`.

---

## 4. Diagnostic method (reproducible)

For each batch, capture a real decode-step trace and sum `comp_time` by canonical layer name:

```bash
/opt/venv/astra-sim/bin/python -m serving \
  --cluster-config configs/cluster/deepseek_r1_single_instance.json \
  --dataset workloads/fixed-8192-4096-b{1,16,64}.jsonl \
  --max-num-seqs 256 --max-num-batched-tokens 4096 --num-reqs {9,24,72} \
  --network-backend analytical --no-cleanup-inputs --log-level INFO
```

- Traces land at `astra-sim/inputs/runs/run_*/trace/H200/deepseek-ai/DeepSeek-R1/instance0_batch<M>.txt`
  (`batch0/1` = prefill chunks, later batches = decode). Pick a decode step at `kv≈8192` (the measured
  request; verify via an `attention_*` line's `input_size = 576·2·kv` per request).
- **MoE** step cost = Σ`expert_*` lines ÷ `ep_size` (ranks run in parallel).
- **Collectives / critical-path residual** = `sim TPOT − Σcompute` (ASTRA-Sim computes the critical
  path; for a serial transformer this ≈ collective latencies — but see the confound in §6).
- The gate's `INFO` log line (`gate_function.py:150`) prints the actual per-rank `local=… activated=…`,
  confirming the `(x,y)` used: **b1 `tokens=1 local=[1…] activated=[1…]`**, **b16 `tokens=16 local=[2…]
  activated=[16…]`**, **b64 `tokens=64 local=[8…] activated=[32…]`**.

---

## 5. Results — per-batch decode-step decomposition

Each column is one decode step's per-NPU critical-path compute (ms), from the trace.

| category | b1 | b16 | b64 |
|---|---|---|---|
| **MoE experts** | **11.80** (70.1%) | **16.63** (70.7%) | **26.56** (66.1%) |
| MLA attention | 1.84 | 3.70 | 10.31 |
| MLA proj + dense FFN | 2.56 | 2.56 | 2.64 |
| norms | 0.54 | 0.54 | 0.57 |
| lm_head / sampler / embed | 0.10 | 0.10 | 0.13 |
| **Σ compute** | **16.84** | **23.53** | **40.21** |
| collectives / residual (TPOT−Σ) | 1.64 | 3.27 | 8.40 († confounded) |
| **sim TPOT** | **18.48** | **26.80** | **48.61** |
| bench TPOT | 9.82 | 20.1 | 39.77 |
| over-prediction | **+88%** | **+33%** | **+22%** |

MoE per-layer charged (all at **covered, profiled** grid points):

| batch | MoE / layer | ×58 (ms) | lookup | note |
|---|---|---|---|---|
| b1  | 203.5 µs | 11.80 | `moe(1,8)` | **FLOOR** — activated clamped 1→8 |
| b16 | 286.7 µs | 16.63 | `moe(2,16)` | activated=16 (= 2·top_k, the boundary) |
| b64 | 458.0 µs | 26.56 | `moe(8,32)` | activated=32 (= E_rank cap) |

**Key clean fact (b1):** `MoE alone (11.80 ms) > bench total (9.82 ms)`. No matter the other terms, an
11.8 ms MoE charge cannot be right when the whole real step is 9.82 ms. Further, the sim's non-MoE
(compute 5.04 + residual 1.64 = **6.68 ms**) `< bench 9.82 ms`, implying **real b1 MoE ≈ 3.14 ms
(~54 µs/layer)** — physical for 1 expert/rank (≈9 µs fp8 weight-load + gate/permute/launch). So at b1 the
non-MoE terms are accurate and the *entire* over-prediction is the MoE floor.

**MLA attention is physically accurate** (b64 10.31 ms ≈ 169 µs/layer = 1.35× its KV-read floor, matching
`next_steps` §15.1.3). It is *not* implicated at any batch.

---

## 6. Coverage verification (answers "is b64 even covered?")

**Yes — every runtime lookup is a genuinely profiled row, nothing extrapolates.**

MoE grid (`profiler/perf/H200/deepseek-ai/DeepSeek-R1/bf16/tp1/moe.csv`, 63 rows), power-of-two on both
axes, `top_k ≤ activated ≤ tokens·top_k`:

```
 tokens | activated_experts present
      1 | [8]
      2 | [8, 16]
      4 | [8, 16, 32]
      8 | [8, 16, 32, 64]
     16 | [8, 16, 32, 64, 128]
     32..4096 | [8, 16, 32, 64, 128, 256]
```
- `moe(1,8)=203.52`, `moe(2,16)=286.658`, `moe(8,16)=291.554`, `moe(8,32)=457.966`, `moe(64,32)=498.765`
  — all **present** (profiled). Activated-axis slope ≈ **10.4 µs/expert**; tokens axis nearly flat at low
  `tokens` (weight/overhead-bound: `moe(1,8)=203.5 → moe(64,8)=237.5`).
- The Fix-A *target* `moe(8,28)` / `moe(2,13)` are **not** rows (power-of-two grid) but `_lookup_moe`
  **bilinear-interpolates** them (verified: it brackets the activated axis and interpolates between rows).

Attention grid (same variant, `attention.csv`):
- `n_decode ∈ {0,1,2,4,8,16,32,64,128,256}` → **n_decode=64 is exact** (nearest-neighbour axis).
- `kv_decode ∈ {…,8192,16384,32768,65536}` → runtime kv 8192→12288 is **bilinear** between 8192/16384.
- `prefill_chunk` densified to 4096 (chunk-factor 1.5).

So the residual is **not** a coverage/extrapolation artifact — the runtime operating points (n_decode ≤
64, kv ≤ 12288, per-rank tokens ≤ 8, activated ≤ 32) sit inside the profiled envelope.

### Profiler runbook (how the profile was produced)
Driver: **`profiler/profile-deepseek-h200.sh`** (working tree). Runbook: **`PODMAN_SETUP.md`** (committed
in `31030ce`, later dropped from the branch; recover with `git show 31030ce:PODMAN_SETUP.md`). Effective
R1 params (match the runtime): `--max-num-batched-tokens 4096`, `--max-num-seqs 256`, `--attention-max-kv
65536` (uniform grid ceiling; skew 32768), `--attention-chunk-factor 1.5`, `--attention-kv-factor 2.0`,
`--measurement-iterations 3`, `--tp 1,8`, `VLLM_USE_FLASHINFER_MOE_FP8=1`, `--hf-overrides
'{"num_hidden_layers":2,"first_k_dense_replace":1}'`, `load_format=dummy` (shapes are real; only the
layer *count* is reduced — the sim multiplies by 58). Profiled dtype is fp8 weights / bf16 activations
(the `bf16/` folder names the *activation* dtype; verified fp8-faithful in `next_steps` §15.1.2).

### The confound at b16/b64 (†)
The `collectives/residual = TPOT − Σcompute` at b16/b64 is **not trustworthy**: the fixed-length **burst**
workload overlaps prefill and decode, so there is no clean matched-kv decode step (early "decode" ITL is
~200 ms — prefill of the 64-burst still draining — and only settles to ~48 ms late, at kv≈12288). The
trace is a single step at kv≈8192–8260 while the TPOT is window-averaged over kv 8192→12288, so the
residual absorbs attention-growth + prefill contention. **b1 is largely immune** (its attention is flat
and overhead-bound), which is why the b1 attribution is clean and the b16/b64 attribution is not.

---

## 7. Attribution and the honest limits of a fix

### b1 (+88%) — the sub-top_k floor. **Proven.**
- **Mechanism:** `route_ep → (1,1)`; `_lookup_moe` clamps `activated 1→8` → `moe(1,8)=203.5µs`, i.e. an
  **8-expert weight-load charged to a rank that touches ~1 expert**, ×58 = 11.8 ms.
- **Proof it's wrong:** MoE alone (11.8 ms) > bench total (9.82 ms). Direction of the fix is certain.

### The `moe(1,1)` problem — why "±2%" was an overclaim
**We do not have, and cannot measure, `moe(1,1)`.** In the profiler every token fires `top_k=8` experts,
so `activated < 8` is structurally impossible (`categories.py:496`; `moe_hook.forge` raises for
`activated < top_k`). So a b1 fix cannot "look up the right value" — there is none. Any fix must **model**
the sub-top_k cost from the grid we *do* have, e.g. decompose:
- **per-expert weight-load** ≈ 10.4 µs/expert (activated-axis slope), and
- **fixed overhead** ≈ `moe(1,8) − 8×10.4 ≈ 120 µs` (gate over 256 experts + permute + launch),

then charge `fixed + activated·per_expert` for `activated<8`, giving `moe(1,1) ≈ 130 µs`.

**But that does not close b1.** `130 µs × 58 = 7.55 ms` → b1 lands at ~14.2 ms vs bench 9.82 ms = **still
~+45%**. To match bench, real b1 MoE must be ~3.14 ms (~54 µs/layer), i.e. the modeled `moe(1,1)` is still
~2.4× too high. The gap is the **~120 µs fixed overhead** — measured with all 256 experts co-located on
one GPU at tp=1 — which a real ep8 rank (owning 32 experts) does not fully pay. This is a **second,
separate defect** (`next_steps` §15.1.1 candidate 3: tp1 gate/permute over-count for EP ranks) that
un-clamping the per-expert axis **does not touch**.

**Corrected verdict for b1:** the floor is provably too high; un-clamping (Fix B) moves it the right way
but likely only from +88% to ~+45%. Pinning b1 accurately needs either a **real per-EP-rank MoE
measurement** (requires a multi-GPU EP profiling mode the profiler lacks — see `next_steps` §16, ~1–2
weeks) or **calibrating** the sub-top_k model to the single bench-b1 number (fits one point, may not
generalize). **I retract the earlier "closes b1 to ±2%."**

### b16 / b64 (+33% / +22%) — activated over-count + confounded remainder
- The MoE lookups are **covered/valid** (`moe(2,16)`, `moe(8,32)`).
- **Fix A** (below) corrects the activated count from the perfect-balance `min(N,32)` to the balls-in-bins
  expectation (b16: 16→~13, b64: 32→~28). Impact is **modest**: `moe(2,16)→moe(2,13)` ≈ −1.7 ms; `moe(8,32)
  →moe(8,28)` ≈ −2.4 ms (~10% of the MoE term). It does **not** close +33%/+22%.
- The remainder is **confounded** (§6) — could be the collective model and/or the same fixed-overhead
  candidate-3 effect. **Not cleanly attributable without a steady-state (non-burst) measurement.**

---

## 8. Candidate fixes (deferred — not applied)

### Fix A — balls-in-bins activated count  ·  `serving/core/gate_function.py:185`
Replace the perfect-balance count with the expected distinct-expert count when `pairs_per_rank`
token-expert pairs land (≈uniform) on the rank's `E_rank` experts:
```python
# current:
activated_per_rank = min(int(round(pairs_per_rank)), E_rank)
# proposed:
activated_per_rank = max(1, min(round(E_rank*(1 - (1 - 1/E_rank)**pairs_per_rank)), E_rank))
```
- Targets b16/b64/ShareGPT. **Modest** (~10% of MoE).
- **~no-op at ep=1** (Qwen3's validated regime) *except* a mid-batch window `pairs ≈ E` — so a **Qwen3
  regression guard is mandatory** (must stay ~±2%).
- Works with the existing bilinear `_lookup_moe` (no lookup change needed).

### Fix B — model the sub-top_k floor  ·  `serving/core/trace_generator.py:844–847` (+ `_lookup_bounds:471`)
Stop clamping `activated<8` to `moe(·,8)`; instead charge `fixed + activated·per_expert` (per §7).
- Targets b1. **Estimate only** — bounded above by the un-verifiable fixed/per-expert split; and it does
  **not** remove the candidate-3 fixed-overhead over-count, so expect a residual (~+45%, not ±0%) unless
  that second term is also addressed.
- Cannot be resolved by re-profiling (structurally impossible at `activated<top_k`).

---

## 9. Open items / what needs data

1. **The b16/b64 remainder is not cleanly attributed** — the burst workload confounds the collective /
   step-latency term. Needs a **clean steady-state (non-burst) decode** measurement at matched kv to
   isolate whether it's the collective model or the candidate-3 fixed-overhead.
2. **`moe(1,1)` and the tp1-fixed-overhead-vs-EP-rank issue** are only resolvable with a **real
   multi-GPU EP MoE profile** (the profiler boots single-GPU tp=1 today). Until then, b1 is
   direction-certain, magnitude-uncertain.
3. **Not in scope here (separate known defects):** KV over-admission hard-crash (b256 / long agentic —
   `memory_model.py:431`); no TP-sharded-experts mode (EP-off can't load — `memory_model.py:156`); PP
   decode hang (`scheduler.py`). See `next_steps_deepseek_r1.md` §15.2.

---

## 10. Bottom line

- **b1 +88% is the MoE sub-top_k floor — confirmed, and it is simulator code, not profile data** (the data
  point that would fix it is unmeasurable, so the fix is a model).
- **b16/b64 look up covered, valid MoE points; attention is accurate.** The activated-count fix is real
  but small; the rest of those two is not yet cleanly explained (burst-workload confound).
- **The honest status of a fix:** Fix B corrects the b1 floor's *direction* but, alone, likely lands
  ~+45% (not ±2%) because of a second, un-addressed tp1-fixed-overhead term; Fix A trims b16/b64 by ~10%.
  Closing either to single digits needs per-EP-rank MoE data or explicit calibration — decisions to make
  before touching `gate_function.py` / `_lookup_moe`.

---

## 11. EP-off cross-validation resolves the confound (2026-08-11)

The user's real-vLLM **EP-off** ground truth for the same workloads (`r1-fixed-noEP-b{1,16,64}` = 12.7 /
22.8 / 48.5 ms; `r1-tp8-sharegpt` = 59.3 ms) supplies the second config that §7/§9 said was needed. EP-off
runs the MoE **TP-sharded** — every rank runs the gate/permute over **all 256 experts** (like the tp1
profile) but at 1/tp width — which is exactly the regime that isolates the fixed-vs-shardable split.

**MoE-cost decomposition (fit from `tp1/moe.csv`, reproduces the grid to ±0%):**
`moe(T,A) = FIXED + WEIGHTLOAD(A) + GEMM(T)` with **FIXED = 120 µs** (gate-256 + permute + launch,
*non-shardable*), **WEIGHTLOAD = 10.4 µs/expert** (*shardable ÷tp*), **GEMM = 0.72 µs/token** (*shardable*).

**EP-off analytical check** (`FIXED + [WEIGHTLOAD(A_cluster)+GEMM]/tp` ×58 + measured non-MoE):
b1 → 12.6 vs 12.7, b16 → 23.6 vs 22.8 (both **within ~1 ms**); b64 → 40.2 vs 48.5 (+8.3 ms residual).
The ring-ALLREDUCE is **~0.1 ms** (bandwidth-tiny) — so the b64 residual is *not* the collective; it is
window-averaged attention (bench median over kv 8192→12288) + TP-shard GEMM inefficiency, which the built
sim resolves. The b1/b16 match **confirms FIXED=120 µs is real in the all-256 regime AND that the non-MoE
compute is accurate** → the **b1 collectives are NOT over-charged** (closes §9.1's open question).

**This splits the EP-on b1 +88% cleanly into two ~4.3 ms terms:**
1. **Sub-top_k weight-load** (Fix B): the clamp charges 8 experts' weight-load (`moe(1,8)`) when a rank
   fires ~1 → un-clamping to `FIXED + 1·WEIGHTLOAD` saves **73 µs/layer = 4.3 ms**.
2. **Fixed-overhead expert-count dependence:** the tp1/EP-off FIXED (120 µs, gate+permute over **256**
   experts) is charged to an ep8 rank whose permute is over its **32** local experts (real ≈ **44 µs**) →
   **76 µs/layer = 4.4 ms**. (This is the former "candidate 3," now *quantified* from the EP-off match, not
   requiring a multi-GPU profile.)
Sum ≈ 8.7 ms ≈ the measured b1 over-prediction (8.66 ms).

**Consequences:** (a) Fix B alone lands b1 ~+45% — term 2 (scale the fixed overhead by local expert count,
ep-dependent) is required to close it. (b) The decomposition is the shared machinery for both the EP-off
TP-sharded latency model *and* EP-on Fix B. See `plans/peaceful-noodling-lantern.md` for the build.

---

## 12. EP-off simulation built + validated (2026-08-12)

The simulator previously **could not run EP-off at all** (ep=1 replicated all 256 experts → 4906 GB OOM at
`memory_model.py:156`). Three small changes made it runnable; all are gated on `ep_total==1 and tp_size>1`,
so **every existing config is byte-untouched** (proven below).

- **Part 1 — weight** (`memory_model.py`): `calculate_sizes('moe', …, moe_tp=max(1, tp//ep))` divides the
  expert intermediate dim by `moe_tp`. EP-off (tp8/ep1) → `moe_tp=8` → 76.6 GiB/GPU (fits); ep=tp → `moe_tp=1`
  (bit-identical).
- **Part 3 — latency** (`trace_generator.py`): `_lookup_moe_tp_sharded = FIXED + (moe(T,A)−FIXED)/moe_tp`
  (the §11 decomposition), where `_moe_fixed_ns` fits FIXED=120 µs from the grid.
- **Part 4 — trace** (`_emit_moe_block`): TP-shard branch emits one `moe` comp line + **ALLREDUCE** over the
  TP dim (the `down_proj` pattern), no dispatch/EXPERT blocks.

**Validation — sim EP-off vs real vLLM EP-off (tp8, `deepseek_r1_tp8_ep1_single_instance.json`):**

| workload | sim TPOT | bench | error | EP-on error (for contrast) |
|---|---|---|---|---|
| fixed b1 | 14.5 ms | 12.7 | **+14%** | +88% |
| fixed b16 | 27.6 ms | 22.8 | **+21%** | +33% |
| fixed b64 | 49.5 ms | 48.5 | **+2%** | +22% |
| ShareGPT (300) | 44.0 ms | 59.3 | **−26%** | +21% |
| SWE-bench (50) | — | 54.2 | KV-over-admission crash (out of scope) | (same) |

**Regression proof (both byte-exact):** EP-on R1 b1 = **18.48 ms**, byte-identical to the pre-change baseline;
Qwen3 example (`single_node_moe_dp_ep_instance.json`) run on HEAD-reverted code vs my-changes code = **byte
identical (21/21 lines)**. So the EP-off edits are a *true no-op* for every ep≥tp / tp=1 config.

**Read of the numbers:** EP-off is *far* better than EP-on everywhere, but the errors are **mixed-sign** — the
tp1-derived model over-charges small fixed batches (+14/+21%) and under-charges large realistic batch (−26%).
That mixed sign is the crux for §14.

---

## 13. What needs fix — the MoE model, in concrete detail

All four defects are one root: **the tp1 `moe.csv` (256 experts, full width, one GPU) is applied as a
per-rank cost.** Each fix below lists *the exact formula change*, *which latency term it moves*, *the
direction*, *which configs it touches*, and *the expected number*.

### Fix A — activated-expert count (`gate_function.py:185`)
- **Now:** `activated_per_rank = min(round(pairs_per_rank), E_rank)` → assumes *perfect* load balance
  (R1 ep8: b16→16, b64→32).
- **Change to:** `round(E_rank·(1 − (1 − 1/E_rank)**pairs_per_rank))` → the *expected distinct* experts a
  rank's pairs actually hit (b16→~13, b64→~28).
- **Moves:** the `activated` axis of `_lookup_moe` → lowers **WEIGHTLOAD** (the dominant, ~10.4 µs/expert axis).
- **Direction: ↓ latency.** **Touches:** EP-on mid-batch (`moe(2,16)→moe(2,13)` etc.) and EP-off A_cluster.
  **~no-op at ep=1 except a mid-batch window** (`pairs≈E`) → **Qwen3 regression guard required** (Qwen3's
  byte-exact check above did *not* include Fix A — it is a separate, still-unapplied change).
- **Expected:** b16 EP-on −1.7 ms (+33%→~+25%), b64 −2.4 ms (+22%→~+16%). Modest.

### Fix B — sub-top_k floor (`_lookup_moe`, the `activated<8` clamp)
- **Now:** `activated<top_k` snaps to the 8-expert row → `moe(1,8)=203.5 µs` at b1.
- **Change to:** for `activated<8`, charge `FIXED + activated·WEIGHTLOAD (+ tokens·GEMM)` instead of clamping →
  `moe(1,1) ≈ 120 + 10.4 = 130 µs`.
- **Moves:** removes 7 experts' worth of WEIGHTLOAD at low batch.
- **Direction: ↓ latency.** **Touches:** EP-on batch<8 only. **No-op for EP-off** (b1 A_cluster=8, no clamp)
  and **Qwen3** (activated≥8). 
- **Expected:** b1 EP-on MoE 11.8→7.6 ms, **b1 +88%→~+45%** (leaves the FIXED term below).

### Fix (2) — FIXED overhead's local-expert-count dependence (the big EP-on lever)
- **The defect:** `FIXED=120 µs` bundles gate + **permute over 256 experts** + launch. A real ep8 rank
  permutes over its **32 local** experts, so its FIXED ≈ **44 µs** (back-solved and EP-off-validated).
- **Why it's hard analytically:** the tp1 profile *cannot* separate the permute-over-256 sub-term from
  gate+launch — they're one measured number. So there's no clean formula on the current CSV.
- **The clean fix = per-rank-shape profiling** (see below): profile MoE with `num_experts = E/ep = 32`, and
  FIXED is *measured* over 32 experts.
- **Direction: ↓ latency.** **Touches:** EP-on (`ep>1`) only. **No-op for EP-off** (permute *is* over 256 —
  that's why EP-off b1 already validated at 12.6). **No-op at ep=1** (E/ep=E). 
- **Expected:** with Fix B already applied, dropping FIXED 120→44 → 58×54 µs ≈ 3.1 ms MoE → **b1 → ~9.8 ms ≈
  bench.** This is the term that actually closes EP-on b1.

### Fix (4) — TP-shard inefficiency (EP-off, and the only ↑-latency fix)
- **The defect:** `shardable/moe_tp` assumes the weight-load+GEMM shard *perfectly* by tp (100% bandwidth
  scaling). Real TP-sharded MoE at large batch is slower (smaller per-rank GEMMs, lower HBM/tensor-core
  utilization). → **under**-predicts (ShareGPT −26%).
- **Change to:** either an efficiency factor `FIXED + shardable/(moe_tp·η)` (η<1, calibrated), or —
  cleanly — per-rank-shape profiling with `moe_intermediate = 2048/tp`.
- **Direction: ↑ latency.** **Touches:** EP-off (`moe_tp>1`) **only** — *never* EP-on or Qwen3.
- **Expected:** raises EP-off large-batch (ShareGPT −26%→toward 0); must be GEMM-size-weighted so it doesn't
  worsen EP-off b1/b16 (whose small +14/+21% is mostly kv-window averaging, not a real over-charge).

### The clean fix that subsumes (2) and (4): per-rank-shape MoE profiling
The profiler already emulates per-rank shapes **single-GPU** for dense/attention (divides hidden/heads by TP
via `hf_overrides`). Extend the same to MoE, keying `moe.csv` by the effective `(tp, ep)`:
- **EP-on**: profile with `num_experts = E/ep` (32) → FIXED measured over the real local expert count → fixes (2).
- **EP-off**: profile with `moe_intermediate = moe_intermediate/tp` (256) → real sharded GEMM/weight-load →
  fixes (4).
Both are **single-GPU** (no 8-GPU run needed). Fix A/B still ride on top for the routing distribution. This is
the item worth scoping — it removes *all* the derivation guesswork and takes both EP-on and EP-off to
validation grade.

---

## 14. Bandwidth, latency direction, and why the fixes don't break Qwen3

Three questions raised together: are we assuming 100% effective bandwidth (⇒ under-predicting memory-bound
layers), do these fixes push latency down or up, and is `link_bw=100` wrong?

### 14a. Where a 100%-effective-bandwidth assumption actually lives
- **Profiled compute (moe, attention, dense) — the *main* memory-bound layers — is MEASURED on real H200**, so
  it already embeds the *real* effective HBM/tensor-core bandwidth. It is **not** idealized and **not**
  under-predicted. This is the key correction to the premise: the heavy memory-bound work is real, not 100%-of-peak.
- **Collectives:** ASTRA-Sim `Congestion_Unaware` uses `link_bw` (per-link GB/s, `Link.cpp:48`) as the
  *effective* value — so effective bandwidth *is* whatever `link_bw` is set to (see 14c). Not a 100%-of-peak
  idealization; a directly-set effective number.
- **`local-mem-bw = npu_mem.mem_bw = 4800`** (H200 HBM peak) feeds only the small `MEM_LOAD`/`MEM_STORE` nodes
  (embedding input, sampler output) — using peak as effective slightly *under*-charges those, but they are a
  <1% term.
- **The one real 100%-scaling assumption is Fix (4)** — `shardable/tp` in the TP-shard model. That is exactly
  where the EP-off under-prediction comes from, and the only place "increase the latency to be more correct"
  applies.

**So: the profiled memory-bound layers are not under-predicted; only the TP-shard *derivation* (4) is.**

### 14b. Do the fixes push latency down or up? (and do they only touch the right error-sign?)

| fix | Δ latency | error it targets | EP-on (+%)? | EP-off? | Qwen3 (near-perfect)? |
|---|---|---|---|---|---|
| A (activated) | ↓ | EP-on over | yes | yes | mid-batch risk → **guard** |
| B (sub-top_k) | ↓ | EP-on b1 over | yes (b1) | no-op | no-op |
| (2) FIXED-local | ↓ | EP-on over | yes | no-op | no-op |
| (4) TP-shard | **↑** | EP-off under | **no** | yes | **no** |

**The answer to "will increasing latency only affect the currently-negative ones and leave the +% ones
alone":** yes, at the config level. **The only ↑-latency fix, (4), lives entirely on the EP-off code path
(`moe_tp>1`) and never executes for EP-on or Qwen3.** So raising EP-off latency to fix the −26% cannot touch
the EP-on +% cases or the near-perfect Qwen3 point. The EP-on over-predictions are corrected *only* by the
↓-latency fixes (A/B/2), which operate on the profiled (real-bandwidth) values by fixing expert-count/routing
— they are **not** bandwidth changes and cannot make a memory-bound layer artificially fast.

**Qwen3 stays near-perfect:** it is ep=1/tp=1 on RTXPRO6000. (4) is EP-off-only; (2) is no-op at ep=1
(E/ep=E); B is no-op (activated≥8). Only **A** can perturb it (mid-batch `pairs≈E`), which is why A carries a
mandatory byte/percent regression guard. Everything else is provably inert for Qwen3 (§12 byte-exact).

### 14c. Is `link_bw=100` (H200) wrong? — **Yes, ~4.5× too low** (code-derived, not pattern-guessed).
Traced through the active `Congestion_Unaware` backend: a message's delay is
`hops × latency + chunk_size / bandwidth` (`BasicTopology.cpp:57-58`), where `bandwidth` = the config
`link_bw` in GB/s; `FullyConnected → hops = 1` (`FullyConnected.cpp:28`); and ALLREDUCE uses the **ring**
algorithm (`config_builder.py:269` → `["ring"]`; `Ring.cc:44` → `stream_count = 2·(N−1)`), i.e. `2(N−1)`
steps each sending `S/N` bytes. So all-reduce time `= 2(N−1)·latency + (2(N−1)/N)·S/link_bw`, and the algebra
gives **effective all-reduce *bus* bandwidth = `link_bw` exactly** (for a ring, busbw = per-link bandwidth).

Therefore `link_bw` must be the **real all-reduce bus bandwidth**, which on 8×H200 over NVLink/NVSwitch is
**~400–480 GB/s** (≈ the 450 GB/s per-*direction* NVLink; the datasheet 900 is the *bidirectional* aggregate,
which NCCL never reports as busbw). Consequences:
- **R1's `link_bw=100` is ~4.5× too low** → it *over*-charges the bandwidth part of every collective.
- The `single_node_single_instance_H100` config's `900` is ~2× too *high* (bidirectional aggregate, not the
  ring's per-direction link) — and its `latency=0` is unphysical. So the configs are inconsistent *and* both
  wrong; the correct value is **~450**.
- My earlier "100 ≈ 900/8 per-link, so it's fine" was **wrong** — the ring makes busbw = `link_bw`, not
  `link_bw × peers`, so there is no `/8`. The datasheet-900 and the per-peer reasoning were both red herrings.

**Impact is bounded** (why it isn't the R1 story): at small batch the collective is *latency*-bound
(`hops×latency`; `link_bw` irrelevant → b1 unaffected); the `S/link_bw` term only bites at large batch, and
even there the collective is a minority of the step. *Empirical cross-check (b64 EP-off decode TPOT):
`link_bw` 100→900 = **49.5 → 46.3 ms (−6%)**; the code-correct **450 → 46.7 ms (−4% vs bench 48.5)**, i.e.
between the two extremes as expected.* **Caution:** `link_bw` is a shared knob across all collectives and both
error-signs — raising it toward the correct ~450 (faster link) helps EP-on over-prediction slightly but
*worsens* the EP-off under-prediction, so it must be corrected **together with** the MoE-model fixes, never in
isolation. **This correction is now applied** (`link_bw 100→450`, commits `a3cfacd`+`cc4f6cd`) and the full
re-measured baseline is **§16** — which also surfaced a *new* prefill-TTFT under-prediction that `link_bw=100`
had been masking.

---

## 15. Recommendation (prioritized)

1. **Fix A + Fix B** — cheap routing corrections, ↓-latency, low risk (A needs the Qwen3 guard). Takes EP-on b1
   +88%→~+45%, b16/b64→~±10%. R1 becomes *usable*, not validation-grade.
2. **Per-rank-shape MoE profiling** — the real fix; single-GPU profiler change, keys `moe.csv` by effective
   `(tp,ep)`. Subsumes (2) [EP-on FIXED-over-local-experts] and (4) [EP-off TP-shard width], taking **both**
   EP-on and EP-off to validation grade. This is the one worth scoping in depth.
3. **Fix `link_bw` to ~450 GB/s** (the H200 all-reduce busbw; code-derived in §14c — busbw = `link_bw`).
   R1's `100` is ~4.5× too low and over-charges collectives; it's a real config bug (and the H100 config's
   `900` is ~2× high). A *minority* term, not the R1 story — but correct it **with** the MoE fixes, since
   raising it in isolation worsens the EP-off under-prediction. Standardize the value across H100/H200 configs.
4. **Out of scope, tracked separately:** KV over-admission crash (SWE-bench / b256, `memory_model.py:431`),
   PP decode hang. Orthogonal to the MoE model.

**One-line summary:** every R1 MoE error — EP-on over, EP-off under — is the tp1 profile used per-rank without
the local-expert-count and TP-width corrections; the profiled compute itself is real-bandwidth and sound, so
the fixes correct *routing/shape*, not bandwidth, and the only latency-*increasing* fix is EP-off-only and
cannot disturb the validated EP-on / Qwen3 points.

---

## 16. link_bw=450 corrected baseline (2026-08-12) — measured BEFORE any MoE fix

`link_bw` corrected `100 → 450` (all DeepSeek H200 single-node configs; `[450,50]` for 2-node; H100 `900→450`;
committed `a3cfacd`+`cc4f6cd`). Full sweep, **no MoE fix**, to give the clean reference. The **link100 column
is freshly re-measured this session** (the plan always intended a side-by-side — the first pass only recorded
`TPOT@100`); **link450 fixed points reproduce the earlier §16 numbers exactly** (b1 spot-check TTFT 255.2 /
TPOT 18.43 ms; every fixed `TPOT@100` reproduced to the decimal), so the two columns share one method.
ShareGPT is re-run fresh at both link_bw (settings pinned: `--max-num-seqs 256 --max-num-batched-tokens
4096`). Fresh EP-on ShareGPT `TPOT@100 = 42.1` (+5%) differs from the earlier column's `48.7` (+21%); every
*other* point — including EP-off ShareGPT (44.0) and **link450** EP-on ShareGPT (38.0) — reproduces the prior
numbers to the decimal, so the EP-on ShareGPT@100 delta is **isolated and unexplained** (most likely a
differing MBT/arrival setting in the earlier ad-hoc run — *not* `f1384c7`, whose fix post-dates the +21%
measurement), flagged for follow-up; the fresh value is used. Bench is real vLLM (link_bw-independent).
*(SWE-bench & b256 excluded — KV-crash.)*

**Table A — TTFT (ms, median):**

| point | bench | sim@100 | err@100 | sim@450 | err@450 |
|---|---|---|---|---|---|
| EP-on b1 | 389 | 413 | +6% | 255 | −34% |
| EP-on b16 | 3505 | 3998 | +14% | 2574 | −27% |
| EP-on b64 | 13208 | 14874 | +13% | 9654 | −27% |
| EP-on ShareGPT | 76 | 67.7 | −11% | 58.2 | −23% |
| EP-off b1 | 538 | 446 | −17% | 265 | −51% |
| EP-off b16 | 4834 | 4288 | −11% | 2654 | −45% |
| EP-off b64 | 18106 | 15944 | −12% | 9950 | −45% |
| EP-off ShareGPT | 118 | 70.3 | −40% | 62.1 | −47% |

**Table B — TPOT (ms, median):**

| point | bench | sim@100 | err@100 | sim@450 | err@450 |
|---|---|---|---|---|---|
| EP-on b1 | 9.8 | 18.5 | +89% | 18.4 | +88% |
| EP-on b16 | 20.1 | 26.8 | +33% | 26.2 | +30% |
| EP-on b64 | 39.8 | 48.6 | +22% | 46.2 | +16% |
| EP-on ShareGPT | 40.1 | 42.1 | +5% | 38.0 | −5% |
| EP-off b1 | 12.7 | 14.5 | +14% | 14.5 | +14% |
| EP-off b16 | 22.8 | 27.6 | +21% | 27.0 | +18% |
| EP-off b64 | 48.5 | 49.5 | +2% | 46.7 | −4% |
| EP-off ShareGPT | 59.3 | 44.0 | −26% | 39.1 | −34% |

**Three findings (link100 vs the code-correct link450):**

1. **Fixed-batch *decode* TPOT is the clean MoE signal — barely moved by `link_bw`.** Low-batch decode
   collectives are *latency*-bound, so link_bw is ~irrelevant: EP-on b1/b16/b64 = **+88 / +30 / +16%**,
   EP-off = **+14 / +18 / −4%**, essentially unchanged from link100 (+89/+33/+22, +14/+21/+2). b64 shifts
   most (bigger messages). These are the honest MoE-model errors the §13 fixes target.

2. **Prefill TTFT under-predicts (−27 to −51% at link450) — and §17 now ROOT-CAUSES it to *compute*, not
   the collective model.** The link_bw=∞ asymptote (collectives free) is still 210 ms, −46% below bench 389,
   so no bandwidth model can close it; `link_bw=100` was accidentally-right only by backfilling ~179 ms of
   missing compute with a 4.5×-inflated bandwidth term. **`link_bw=450` is vindicated.** New in the table:
   EP-off prefill under-predicts **even at link100** (−11 to −17%) — its larger (TP-shard) compute deficit
   can't be backfilled — whereas EP-on was +6 to +14% at link100. See **§17**.

3. **ShareGPT (closed-loop) is closer on decode than the earlier numbers implied.** Fresh EP-on TPOT
   **+5% → −5%** (the prior +21%/48.7 is unreconciled — see intro) — realistic-workload decode was never
   badly over-predicted. EP-off TPOT **−26% → −34%** (correct link exposes the MoE TP-shard under-charge,
   §13 Fix 4). ShareGPT *prefill* TTFT under-predicts too (EP-on −11%, EP-off −40% at link100), same compute
   cause as finding 2.

**Net:** the corrected link_bw=450 disentangles three now-independent error sources: **decode TPOT** isolates
the MoE routing model (finding 1 → §13 Fix A/B/2), **prefill TTFT** is a multi-GPU *compute* under-count
(finding 2 → **§17**, NOT the collective model), and **realistic ShareGPT** is close on decode / under on
prefill (finding 3). **Decision point:** the §13 decode MoE fixes still apply unchanged (judge them on
**TPOT**; TTFT stays negative until a separate prefill-compute fix); prefill is its own workstream (§17);
`link_bw` stays 450.

---

## 17. Prefill-TTFT under-prediction — root-caused to compute, NOT the collective model (2026-08-12)

Finding 2 of §16 flagged a prefill-TTFT under-prediction (−27 to −51%) exposed by the link_bw=450
correction, with two candidate causes: (i) large-message collective effective-bw, or (ii) profiled prefill
compute. **This pass resolves it decisively to (ii): the link-independent per-rank *compute* term, and
specifically R1's *multi-GPU* compute — the collective model at link_bw=450 is vindicated.**

### 17a. Method
Single 8192-in request, output truncated to 8 (prefill is output-length-invariant here — probe TTFT
**255.2 ms == §16 EP-on b1 255 ms**, and all fixed measured requests share one arrival time so there is no
decode-length-dependent queuing). The ASTRA-Sim controller log reports, per prefill chunk, the cumulative
cycle count **and** the *exposed communication*, so each chunk splits into compute vs non-overlapped
collective with no modeling. link_bw was then swept to separate the link-independent (compute + latency
floor) term from the bandwidth term.

### 17b. The decisive experiment — sweep link_bw, extrapolate to ∞
EP-on single-8192 prefill TTFT (ms):

| link_bw | 100 | 450 (code-correct) | 900 | ∞ | **bench** |
|---|---|---|---|---|---|
| sim TTFT | 413.4 | 255.2 | 232.6 | **210.0** | **389** |

Fits **`TTFT = 210.0 + 20340/link_bw`** to the decimal (450→255.2, 900→232.6). Hence:
- **C = 210 ms** link-independent (profiled compute ≈191 + collective latency-floor ≈19).
- collective **bandwidth** term = `20340/link_bw` = **45 ms @450**, 203 ms @100, ≈0 @∞.

**At link_bw=∞ — collectives made free — sim TTFT is still 210 ms, −46% below bench 389 ms.** No collective
model can close a gap that survives infinite bandwidth. Therefore the prefill deficit is the **compute**
term, and **link_bw=100 was accidentally right** (413 ≈ 389) only because its 4.5×-inflated bandwidth term
(203 ms) numerically backfilled ≈179 ms of missing compute. **The code-derived link_bw=450 (§14c) is
correct; it merely unmasked a pre-existing compute under-count.**

### 17c. Per-chunk decomposition (trace == ASTRA-Sim, validated)
chunk0 (kv=0) compute 86.5 ms, chunk1 (kv=4096) 104.4 ms (Δ = attention 12.3→30.2). Per chunk: **moe_experts
46.8** (largest), mla_proj 13.1, attention 12.3/30.2, norms 7.1, o_proj 6.0, rest ≈1.7. Collective volume is
physically correct (64×58.7 MB ALLREDUCE + 58×(7.6 MB ALLGATHER + 58.7 MB REDUCESCATTER)); at busbw 450 it
is a minority (45 ms bw + ≈19 ms floor).

### 17d. It is R1's narrow-per-rank compute, not a general profiler-sum flaw
**Qwen3-30B (tp1, ep2 — dense/attention at FULL width on one GPU; only a 2-way MoE all-to-all) prefill
validates**: TTFT mean −1.5%, P90 +3.7%, P95 −3.2%, P99 +4.7%; TPOT/latency within ±3%
(`bench/examples/.../validation/rerun_check_summary`). *(Correction: Qwen3 is ep2, not the tp1/ep1 stated in
an earlier draft — but its `tp_size=1` means dense/attention are profiled and run at full width, and the MoE
collective is a small 2-way exchange, so the point stands.)* So the profiled per-layer compute *sum* is
accurate **at full width**. The R1 under-count is therefore specific to the **TP8 narrow-width per-rank proxy**
(`hidden/8, heads/8`, `E/ep` experts) that Qwen3 never exercises — and since collectives are excluded (17b),
it is the per-rank *compute* proxy, not communication (leading candidate: 17f.2).

### 17e. EP-off is worse — the TP-shard MoE model (ties to §13 Fix 4)
EP-off single-8192 prefill: sim 264.2 ms @450, **212.3 ms @∞**, bench **538 ms** → **−61% even at ∞**. The
EP-on and EP-off compute asymptotes are ≈equal (210 vs 212) while the *real* values differ by 149 ms, so the
extra EP-off deficit is the **TP-shard MoE model's perfect-÷8 assumption** (`_lookup_moe_tp_sharded`,
`shardable/moe_tp`) — exactly §13 **Fix 4** (TP-shard inefficiency), now confirmed on prefill.

### 17f. Mechanism hypotheses (WHY the multi-GPU per-rank compute under-counts) — for the fix phase
1. **Sustained-prefill clock/thermal throttling** over 61 layers × 8192 tok, missed by the short profiler run
   (`num_hidden_layers=2`, dummy weights) — scale-dependent, would explain R1 (671B) vs Qwen3 (30B).
2. **TP8 per-rank proxy optimism** — single-GPU `hidden/8, heads/8` emulation vs a real TP8 rank inside an
   8-GPU forward.
3. **MoE EP-rank proxy optimism** (`moe(512,32)@tp1`). *(Note: §13 Fix 2 pushes this DOWN → would slightly
   WORSEN prefill; the two must be weighed together.)*

### 17g. Bottom line + relation to the MoE fixes
Prefill under-prediction is a **new, mostly-independent compute defect**, *not* the collective model and
*not* the decode MoE floor. The decode MoE fixes barely touch prefill (Fix A saturates at 512 tok/rank, Fix B
is an activated≥8 no-op; only Fix 2 nudges it, small and the wrong way; Fix 4 helps EP-off prefill). So: judge
the decode fixes on **TPOT**, and track **TTFT** against a *separate* prefill-compute fix. link_bw stays 450.

---

## 18. Applied Fix A + Fix B — Fix B kept, Fix A reverted; Fix 2 blocked on profiling (2026-08-12)

Applied Fix A (balls-in-bins activated) + Fix B (sub-top_k floor), measured TTFT **and** TPOT vs the §16
baseline with a Qwen3 (ep2) guard, then settled on **Fix B only** (Fix A reverted as unsound). Two measured
passes — post-A+B (both) and the current Fix-B-only state — shown side by side (columns mirror §16).

**Table A — TTFT (ms, median):**

| point | bench | baseline §16 (err) | post A+B (err) | Fix B only — current (err) |
|---|---|---|---|---|
| EP-on b1 | 389 | 255 (−34%) | 255 (−34%) | 255 (−34%) |
| EP-on b16 | 3505 | 2574 (−27%) | 2574 (−27%) | 2574 (−27%) |
| EP-on b64 | 13208 | 9654 (−27%) | 9654 (−27%) | 9654 (−27%) |
| EP-on ShareGPT | 76 | 58.2 (−23%) | 56.0 (−26%) | 58.3 (−23%) |
| EP-off b1 | 538 | 265 (−51%) | 265 (−51%) | 265 (−51%) |
| EP-off b16 | 4834 | 2654 (−45%) | 2654 (−45%) | 2654 (−45%) |
| EP-off b64 | 18106 | 9950 (−45%) | 9950 (−45%) | 9950 (−45%) |
| EP-off ShareGPT | 118 | 62.1 (−47%) | 59.9 (−49%) | 62.1 (−47%) |

TTFT is **unchanged** across all three columns for the fixed points — the fixes are decode-only (at prefill the
per-rank token count saturates `activated` at 32, so both are no-ops); ShareGPT wobbles only via closed-loop
batching feedback. Prefill under-prediction stays the separate §17 problem.

**Table B — TPOT (ms, median):**

| point | bench | baseline §16 (err) | post A+B (err) | Fix B only — current (err) |
|---|---|---|---|---|
| EP-on b1 | 9.8 | 18.4 (+88%) | 14.3 (+46%) | **14.3 (+46%)** |
| EP-on b16 | 20.1 | 26.2 (+30%) | 24.5 (+22%) | 26.2 (+30%) |
| EP-on b64 | 39.8 | 46.2 (+16%) | 43.8 (+10%) | 46.2 (+16%) |
| EP-on ShareGPT | 40.1 | 38.0 (−5%) | 36.6 (−9%) | 38.0 (−5%) |
| EP-off b1 | 12.7 | 14.5 (+14%) | 14.5 (+14%) | 14.5 (+14%) |
| EP-off b16 | 22.8 | 27.0 (+18%) | 24.8 (+9%) | 27.0 (+18%) |
| EP-off b64 | 48.5 | 46.7 (−4%) | 44.2 (−9%) | 46.7 (−4%) |
| EP-off ShareGPT | 59.3 | 39.1 (−34%) | 37.7 (−36%) | 39.1 (−34%) |
| **Qwen3 (ep2)** | 48.9 | 49.6 (+1.4%) | 47.4 (−3.1%) | **47.9 (−2.1%)** |

In the current state **Fix B's only material effects are R1 EP-on b1 (+88→+46%) and Qwen3 (+1.4→−2.1%)** — both
TPOT; b16/b64 and all EP-off return to baseline once Fix A is reverted (Fix B is a genuine no-op there,
`activated ≥ 8`).

**Fix B — KEPT, but NOT inert for Qwen3 (correcting an earlier claim in this doc).** It is physically correct
(don't bill a rank for more experts than it fires) and fires wherever a rank's `activated < top_k` — i.e. **any
ep>1 config at low batch**, including **Qwen3 ep2 at N=1** (`activated = min(round(N·8/2), 64) = 4 < 8`). So it
moves Qwen3 +1.4%→−2.1%. This is *not* a Fix B bug: Qwen3's +1.4% was itself a **lucky cancellation** — the
(wrong) sub-top_k over-charge was compensating a latent ~2% *under*-prediction (the same multi-GPU compute
under-count as §17). Fix B removes the false credit and unmasks it (the link_bw=100 pattern again). Net: a
large, clearly-correct R1 b1 gain for a marginal, honest Qwen3 regression. The residual +46% on b1 is the
FIXED-over-local term (Fix 2).

**Fix A — REVERTED** (`gate_function.py` restored to HEAD). The Qwen3 regression is the smoking gun:
balls-in-bins (uniform-random) *under*-counts activated experts for a **trained, load-balanced** router
(real routers spread tokens far more evenly — near perfect-balance, the old formula). Fix A's R1 b16/b64
gain was **"right answer, wrong reason"** — it coincidentally offset the real b16/b64 defect (Fix 2), and
keeping it would **double-count** once Fix 2 lands (already visible as it pushes EP-off b64/ShareGPT *further*
under, −4→−9% / −34→−36%).

**Fix 2 (FIXED-over-local-experts) — blocked; NOT derivable from the tp1 profile:**
- One FIXED point (over all E experts); splitting the expert-scaling permute sub-term from gate+launch is
  underdetermined (§13). No analytical `FIXED(E_local)` from the current CSV.
- **Correction to §13/§14b:** they said Fix 2 is a "no-op for Qwen3 (ep=1)". Qwen3 is actually **ep=2**
  (E=128 → E_local=64), so Fix 2 *would* reduce Qwen3's FIXED → the same regression risk as Fix A. **Not**
  safe-by-construction.
- Clean fix = **per-rank-shape MoE profiling** (measure FIXED at E_local experts), §15 item 2 — needs the
  vLLM profiler env (GPU + vLLM v0.19.0), not runnable from the ASTRA-Sim container.
- Interim estimate (R1-calibrated FIXED 120→44µs, §11 b1 back-solve): EP-on b1/b16/b64 → **+1/+8/+5%**, but
  R1-specific (params fit to R1, can't transfer to Qwen3) → an interim, not a general fix.

**Current clean state:** Fix B kept + Fix A reverted → EP-on b1 **+46%** (from +88%); b16/b64 **+30/+16%** and
all EP-off at baseline (Fix B no-op there — Fix-2 territory); **Qwen3 −2.1%** (Fix B unmasked a latent
under-prediction). link_bw stays 450; the EP-off build + Fix B remain uncommitted. The clean Fix 2 / Fix 4 =
**per-rank-shape MoE profiling** — code + runbook in `MOE_PER_RANK_PROFILING.md`, runnable once a GPU is
available (the profiler needs the vLLM env; the simulator consumes the keyed `moe.csv` with a byte-identical
fallback to the tp1 profile when absent).
