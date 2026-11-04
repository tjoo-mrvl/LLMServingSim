---
sidebar_position: 5
title: Native MoE components
---

# Native MoE components

DP+EP does not apply one token count to an entire MoE block. Gate and routing
operate on local rows, experts operate on gathered rows, and finalization returns
to local rows. Native component profiling measures these GPU regions separately
while leaving communication to the network model.
The unquantized modular DP1 path measures local gate/routing and experts, with
verified no-op preparation/finalization and no DP dispatch/combine. A separate
capability-admitted live path retains native processed expert weights and backend
at TP1/DP1/EP1, including supported quantized and shared-expert execution. It adds
a local `wrapper` component for the surrounding native work.
FP8 experts of an EP > 1 target at DP1 use the modular path too, with seeded
weights in the checkpoint's FP8 layout and the kernel vLLM selects for that
target; see [FP8 experts on the modular path](#fp8-experts-on-the-modular-path).

The shipped RTXPRO6000 BF16 Qwen3-30B bundle includes a TP1/DP2/EP2 component
entry in its root metadata. Its contracts define the actual bounds and backend; the same model name
at another TP or DP degree does not imply coverage. Published bundles contain
the runtime files, not the original per-forward acquisition streams. Keep the
original acquisition directory to resume it, or choose a separate `--out-root`
for a new measurement.

The adapter uses version-sensitive vLLM internal APIs. Deferred finalization
is checked through `use_deferred_moe_finalize`, the effective boolean property
in 0.28 and 0.31. The similarly named `defer_moe_finalize` is a request flag
in 0.28 and a request method in 0.31; its truthiness is not an execution mode.
Unknown capability APIs and actually deferred execution are rejected.
A vLLM upgrade or a different expert backend requires auditing the whole
execution contract, not merely reusing the same CSV column names. Compatibility
depends on native APIs, geometry, backend and conditioning semantics, not source
hashes, exact library versions or directory names. A semantic method change does
not relabel historical measurements.

## Acquire a deployment-matched table

For a supported unquantized modular deployment, refresh only the MoE category:

```bash
python -m profiler slice Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --hardware RTXPRO6000 --tp 1 --tp-refresh 1 --group moe --dp 2 \
  --max-num-batched-tokens 2048 --max-num-seqs 128 \
  --moe-rounds 3 --measurement-iterations 3
```

`--dp` accepts comma-separated target degrees. EP is resolved as `TP * DP`.
Use `--dp 1` for a local target or `--dp 1,2` to acquire both independently.
DP1 with TP1 is a single-GPU MoE block. At TP greater than one, this option
means whole experts distributed over EP=TP, not tensor-sharded experts.
Do not reuse one geometry's measurements for the other. Whole-block acquisition
and its `--moe-ep-degrees` option have been removed. To refresh TP2,
use `--tp 1,2 --tp-refresh 2`; the existing CLI still requires TP1 in its
session list, but a slice measures only the selected degree.

`--ep` accepts an optional comma-separated filter over those full-EP targets.
Omit it to acquire every TP/DP combination. For example,
`--tp 1,2 --dp 1,2 --ep 2` in a full profile acquires MoE components for TP1/DP2
and TP2/DP1, both EP2; ordinary categories still run at TP1 and TP2.
For a TP2-only MoE slice, use `--tp 1,2 --tp-refresh 2 --dp 1 --ep 2`.
Every explicit EP degree must match the selected TP times DP; unmatched degrees
fail before engine startup. EP selection does not infer additional TP/DP values
or enable tensor-sharded experts. Both shell templates forward `TP_DEGREES`,
`DP_DEGREES` and `EP_DEGREES` as `--tp`, `--dp` and `--ep`.

By default one process measures every rank of each target in turn, on one GPU.
`--moe-ranks` (comma-separated, 0-based) restricts a MoE slice to those ranks.
This lets one process per GPU acquire a target in parallel:

```bash
for rank in 0 1; do
  CUDA_VISIBLE_DEVICES=$rank python -m profiler slice Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --hardware RTXPRO6000 --tp 1,2 --tp-refresh 2 --group moe --dp 1 --ep 2 \
    --moe-ranks $rank > moe-rank$rank.log 2>&1 &
done
wait
```

`profiler/profile-moe-ranks.sh` runs that loop for any MoE model and target:
`MODEL=... HARDWARE=... TP=2 ./profiler/profile-moe-ranks.sh`, with `DP`
(default 1), `OUT_ROOT`, `MOE_ROUNDS` and `MOE_GRID_FACTOR`, and further slice
flags after the script name. It runs one process per visible GPU; with fewer
GPUs than ranks, each GPU measures several ranks in turn.

Each rank keeps its own table folder and writer lock, and every process
replaces only its own entry in the locked `meta.yaml`. Processes with disjoint
ranks can therefore share one output root. Give each process its own GPU.
Processes sharing a GPU disturb each other's timings, and nothing rejects the
result: each point is timed once and published as measured. Overlapping ranks
collide on the lock, and the second writer fails. A rank outside every selected
target fails before engine startup. `profile` does not accept the option, because its
ordinary categories need one writer per variant root.

When every rank of a deployment is published, the process that finishes it logs
a rank cross-check. For each rank, it reports the median, 5th-percentile,
95th-percentile and worst ratio of its time to the median of the other ranks'
times at the same points. Every rank of a full-EP target does the same work at
a point, so a rank whose ratios drift was measured under different conditions,
such as a busy or slower GPU. The check only reports: it changes no data, and
re-measuring a rank is a separate decision. In parallel acquisition, the last
process to finish reports. Earlier processes log which ranks they are still
waiting for.

The profiler uses **one physical GPU**, not TP times DP processes. It retains
the checkpoint's global expert count and top-k, resolves each target rank's
expert map through vLLM, and builds local expert weights in the native layout.
The modular adapter initializes synthetic expert weights. The live local adapter
keeps the engine's dummy-loaded, natively processed weights, quantization scales,
activation settings and selected backend; it does not substitute an unquantized
kernel. Neither path measures a trained router's distribution. Collective execution
and CPU time are excluded. Target and acquisition parallelism are recorded
separately, so a TP2/DP2 table is not evidence of a four-GPU communication run.

Omitting `--dp` selects DP1 for MoE acquisition. The component path also works
in `profile`, replacing only its MoE category; dense models and non-MoE slices
do not acquire MoE components. A MoE slice
does not restamp the main engine, attention grid or skew metadata.

The isolated component engine instantiates the smallest MLP prefix and preserves
the requested token, sequence and context limits. Its private
`kv_cache_memory_bytes` override defaults to a 1 GiB resource cap, leaving space
for independently conditioned expert banks. This is not a capacity guarantee:
vLLM must validate the configured context against its native cache layout, or
startup fails. The internal acquisition caller can raise the cap; context limits
are never reduced to fit it. Actual component-engine settings are observations
in that MoE contract, not replacement cache facts for ordinary profiling or serving.

## Live local boundaries

The live adapter admits actual APIs, tensor layouts and capabilities rather than
model-name aliases. It keeps gate projection and external routing in
`gate_routing`; when the native expert kernel fuses routing, this component holds
projection only and `experts` includes routing. Native expert preparation,
quantization and finalization stay inside `experts`.

`wrapper` runs the original parent MoE forward and retains its CUDA activities
after excluding the gate/router and routed-kernel scopes. Shared experts, latent
down/up projections, normalization, scaling and output combination remain in this
boundary where the native wrapper performs them. Launch ownership and conservation
checks prevent those activities from also entering another component. The native
method and each isolated/cloned expert bank must produce exactly equal outputs
for every routing stimulus before timing; unsupported routing replay or state
changes are errors.
State checks cover parameter bindings/layouts, available version counters,
effective MoE configuration and callable identities. They are not bytewise
checksums of device memory and do not detect arbitrary in-place CUDA writes.

The stored metric is the **sum of unique native CUDA activity durations**, not an
interval union or elapsed critical-path latency. Concurrent gate, routed and
shared branches can overlap in the real model; adding these rows in a serial
trace does not reproduce that overlap. CPU time and launch gaps remain excluded.
These boundaries provide measured work-duration replay, not a new full-model
serving-accuracy claim.

The contract separates original/gate width from the physical expert width and
records the actual gate dtype, including FP32 routers. `expert_weight_bytes`
counts unique native weight/scale storage; `expert_bank_bytes` additionally counts
native state that a conditioned clone must own. The bank allocation guard uses
the latter plus native workspace and reserved free memory. It does not apply the
unquantized adapter's conversion-buffer estimate to already processed weights.
Cloning preserves storage aliases, offsets and strides. Wrapper weight storage
excludes the gate and routed expert storage rather than double-counting them.

## FP8 experts on the modular path

An FP8 checkpoint's EP1 engine runs the expert kernel vLLM selects for EP1, but an
EP > 1 deployment of the same checkpoint can select another: for block-scaled
FP8 on Hopper, Triton at EP1 and FlashInfer CUTLASS above it. The live adapter
would time the EP1 kernel, so an EP > 1 target whose routed experts use vLLM's
`Fp8MoEMethod` is measured on the modular path instead:

- **Weights.** The modular adapter's seeded values (`randn × 0.01`, seed 0),
  stored as the checkpoint stores them: FP8 values with one float32
  dequantization scale per `weight_block_size` block, written by vLLM's own
  `per_block_cast_to_fp8`. A checkpoint declaring `scale_fmt: ue8m0` gets
  power-of-two scales.
- **Kernel.** vLLM's FP8 oracle selects the backend from the target
  configuration, converts the checkpoint layout into the kernel's and builds
  the quantization config, as `Fp8MoEMethod` does. Whichever modular backend it
  selects is accepted; one that needs layer state the adapter does not provide
  is refused by name.
- **Output check.** Once, before any timing, one kernel call is compared with
  an fp32 computation of the rank's own experts over the same dequantized
  weights. The relative Frobenius error must stay below 0.1: the FP8 rounding of
  the two activation quantizations is expected to account for a few percent,
  while a wrong layout, scale or expert map is of order one. A mismatch stops
  the acquisition.
- **Prepare.** vLLM's DP1 prepare quantizes the input to FP8, or hands it to
  experts that quantize it themselves (FlashInfer CUTLASS). Either way it runs
  inside the timed `experts` call, because the deployment runs it there;
  `local_execution.prepare` records `timed_input_quantization` or `identity`.
  Finalize must still be a CUDA-free no-op on the output alias.
- **Weight banks.** Every conditioning bank is rebuilt from its own copy of the
  checkpoint, scales included, and `expert_weight_bytes` counts the FP8 values
  and their scales.

An EP1 target keeps the live path, whose engine already runs the target's
kernel. Refused for now: per-tensor and static-activation FP8 schemes, other
quantization methods and FP8 at DP > 1 (its prepare quantizes before the
dispatch all-gather, which this region does not time). Serving loads both
preparations, because the input quantization is already inside the timed
`experts` call. The label must match `expert_region.input_quantization`:
`experts` with `identity`, and `prepare` with `timed_input_quantization`.
Serving refuses quantized tables at DP > 1.

### Expert formats

Each format the modular path supports is one class in
`profiler/core/hooks/moe_quant.py`: unquantized and FP8 today. A format mirrors
vLLM 0.31's own per-format steps in `fused_moe/oracle/` (select the backend,
convert the checkpoint layout, build the quantization config) and supplies the
seeded checkpoint, its fp32 dequantization for the output check, its contract
fields and its DP1 prepare step, plus the linear methods a rebuilt shared expert
may use. The format is chosen by the routed experts' own vLLM quant method, not
the checkpoint's quantization config: experts a quantized checkpoint leaves
unquantized (`Fp8Config`'s ignore list gives them `UnquantizedFusedMoEMethod`)
take the unquantized modular path. A quant method no format matches keeps the
live path. The FP8 dtype is the platform's (`current_platform.fp8_dtype()`), as
vLLM's own weight casting writes it.

## Shared experts on the modular path

A model with a shared expert, such as DeepSeek-V3/R1, runs it beside the
routed experts. `MoERunner.forward` then applies the routed scale when the
model applies `routed_scaling_factor` there, adds the two outputs and reduces
the sum over TP once. The shared expert is an ordinary tensor-parallel MLP. In
a full-EP deployment at TP > 1, each GPU holds 1/TP of the shared expert's
intermediate width but owns whole routed experts.

An EP > 1 target measures it as one more component, `shared_experts`, placed
after the others and timed on the rank's local rows:

- **Copy.** The acquisition engine runs at TP1, so its shared expert has the
  full width. The adapter copies the runner's module (`runner._shared_experts`)
  and rebuilds both linears at width/TP with the same vLLM classes,
  quantization config and prefix. vLLM then selects the quantization method and
  GEMM kernel for a TP rank's shapes. The module's own `forward` and `act_fn`
  run unchanged.
- **Admission by structure.** The module must hold exactly these submodules:
  - a `gate_up_proj` `MergedColumnParallelLinear` of two equal halves over the
    hidden size;
  - an `act_fn` without weights;
  - a `down_proj` `RowParallelLinear` over one half that leaves the reduction
    to the runner.

  Neither linear may have a bias or be replicated across TP, and each must be
  unquantized or block-FP8 with dynamic activations. A gated shared expert and
  sequence-parallel MoE, which replicates the shared expert, are refused. Every
  refusal names the mismatch, and the code names no model family.
- **Weights and check.** Seeded values (`randn × 0.01`) are stored in the
  checkpoint layout: gate and up as separate shards, and FP8 with block scales
  where the checkpoint is quantized. They go through the layers' own vLLM
  weight loaders and post-load processing. Before any timing, one call is
  compared with fp32 arithmetic over the same dequantized weights. The relative
  Frobenius error must stay below 0.1.
- **Timed call.** It covers the copy's forward, the runner's native
  `_maybe_apply_routed_scale_to_output`, and the add of the routed output. The
  routed output itself belongs to the `experts` component, so a zero tensor
  stands in for it. The TP reduction of the sum remains a simulated collective.
- **Contract.** The `shared_experts` entry records:
  - the module, activation and linear classes;
  - each linear's quantization method and selected kernel;
  - both widths and the checkpoint weight bytes;
  - `routed_scaling_factor`.

  `excluded` no longer lists `shared_experts`. A model without a shared expert
  keeps its contract and components unchanged.

Like `gate_routing`, this component is measured warm in isolation. Repeated
calls can keep its weights in L2, which does not happen in a full forward. An
EP1 target keeps the live path, whose `wrapper` component already contains the
shared expert.

Serving charges `shared_experts` on the rank's local rows after the routed
experts, and after `finalize_copy` at DP > 1. The TP restore of the sum moves
after it. Its loader checks the boundary the same way for every model: the
width times TP, the TP degree, dtypes, and weight bytes that cover both linears.
The charge is serial. Where vLLM overlaps the shared expert with routed work,
the charge is an upper bound. vLLM overlaps it on an auxiliary CUDA stream for
batches up to `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` tokens, or inside a
modular kernel that supports it. That overlap is not modelled.

## Shapes and measurement protocol

`MoeTarget` is shared by acquisition and serving. Given already-padded DP
counts `n_i`, a non-sequence-parallel rank uses `n_i` local rows and
`sum(n_i)` expert rows. A sequence-parallel rank uses `ceil(n_i / TP)` local
rows and `TP * sum(ceil(n_i / TP))` expert rows. The catalog declares whether
the audited model wrapper uses sequence parallelism; TP alone does not decide it.
This adapter's DP1 path is non-sequence-parallel: local and expert row counts
are equal. The native backend is selected from each target's resolved configuration;
it may differ between DP1 and DP+EP even on the same model and GPU.

For the unquantized modular DP1 path, native preparation must preserve the input without scales or dispatch
metadata. Experts write to the caller's output buffer; the admitted native
NoOP reduction then needs no copy. Acquisition verifies alias identity and
absence of CUDA activity in prepare/finalize before recording these boundaries
as `local_execution` in the root contract. No zero-time or synthetic-copy row
is added. A backend requiring additional work is rejected rather than silently
omitted or replaced by another backend.

`--moe-grid-factor` (default **2**) controls the geometric grids. Local component
rows range up to the local budget; expert input rows reach the derived gathered
maximum. Expert measurements have three independent coordinates:

| Axis | Meaning |
| --- | --- |
| N (`tokens`) | Rows presented to the expert kernel, including remote choices |
| A (`activated_experts`) | Distinct active experts owned by this rank |
| M (`local_assignments`) | Local token-expert pairs; not unique dispatched tokens |

Every point retains the checkpoint's global top-k `k`. For local expert count
`L` and global expert count `E`, positive-work points satisfy
`max(A, N * max(0, k - (E - L))) <= M <= N * min(k, A)`.
The grid includes feasible endpoints and slope-change corners as well as the
balanced assignment total wherever feasible. All-remote `A=M=0` is measured
separately when remote experts can satisfy top-k. At EP1, `M=N*k`; the third
axis therefore collapses without redundant measurements. The grid derives from
configuration and geometry, not benchmark requests or measured latencies.

New components are measured **eagerly only**, with no isolated CUDA graph
creation or replay. Launch-correlated
CUDA activities are attributed to individual forwards without adding CPU
durations or launch gaps. `--moe-rounds` controls independent measurement
contexts; `--measurement-iterations` is the minimum forwards per context.
Expert measurements may use more forwards to complete the required weight cycles.
The stored time is the median of the per-context forward medians, in microseconds.

Expert measurements rotate local expert positions and independent weight banks.
Bank count is the smallest power of two, at least four, whose weights hold four
times the GPU's L2 capacity (one bank for the all-remote point). Warmup and the
timed forwards cover whole rotation cycles, and each point is timed once: the
median of its per-context forward medians is the stored time. No comparison of
timings withholds a table. GPUs are not deterministic, so a pass/fail check on
noisy timings would refuse tables that are as good as any other; judge a table
by the rank cross-check and by the per-forward values kept in `samples.jsonl`.
Independent weight-bank outputs are compared against cloned eager
references, including DP1 output alias checks.
The live local path instead retains native output allocation and checks against
the exact native method at each routing stimulus, including every cloned bank.

These are explicit measurement-conditioning rules, not a guarantee of exact
full-model cache behavior. Local gate/finalization regions remain isolated-warm.
Kernel selection and arbitrary trained routing distributions can still differ.
Different within-rank histograms can share N/A/M: the table approximates them
with a balanced histogram over A active experts, not an exact learned router.

Serving CUDA graph padding remains active. New eager timings are queried at
the already-resolved padded rows; historical graph rows are not relabelled.
The acquisition token axes prefer target serving capture tokens G. Each G is
mapped through `target.token_domains((G,) * DP)` to obtain
local gate/finalization rows and gathered expert rows. The graph resolver handles
compiler SP; the token-domain transformation handles model-level SP. MRV1's
compiler-SP rounding also increases the physical acquisition bound when the
logical token budget is not divisible by TP.

For `FULL`, `PIECEWISE` and `FULL_AND_PIECEWISE`, capture points replace interior
geometric token points in the covered range. Beyond the capture cap, geometric
points and the transition endpoint are included. `NONE` and `FULL_DECODE_ONLY`
retain the full geometric range because prefill/mixed can run eagerly. With DP,
one out-of-capture member can cancel graph replay for small peers: the planner
retains their local geometric range and the possible unequal gathered-row range.
Endpoints and feasible slope-change corners remain even if not capture sizes.
Consequently a custom capture list can remove redundant points, but mandatory
transition/geometry points mean the total is not always smaller.

Capture sizes alone never determine A or M, whose independent geometric axes and
feasible corners are retained. Matching completed worker
captures are preferred unless target options override them; otherwise candidate
captures are derived and labelled as such. Eager measurement at these coordinates
does not capture or replay a graph. See [sampling policy](./running#sampling-policy).

## Files, resume and lookup

The variant-root `meta.yaml::per_tp[tp].moe_components` record indexes target
tables. New output uses readable deployment directories rather than contract hashes:

```text
<variant>/
├── meta.yaml
└── tp<N>/
    └── moe_components/
        └── dp2_ep2_rank0_sp0/
            ├── components.csv
            └── samples.jsonl    # optional raw acquisition checkpoint
```

`components.csv` stores `component,mode,tokens,activated_experts,local_assignments,time_us`.
Modular DP+EP stores `gate_routing`, `experts` and `finalize_copy`; modular DP1
stores `gate_routing` and `experts`. The live local path stores `gate_routing`,
`experts` and `wrapper`. The admitted contract declares the expected component set.
The root record has `schema: moe-components` and `entries`, each containing
`target`, a TP-relative table `path`, `contract` and `coverage`. The contract
describes geometry, target TP/DP/EP/rank, expert ownership, dtypes, native backend
and measurement conditioning, with acquisition observations kept as provenance.
New contracts record `modes: [eager]` and `expert_sampling` (axis names, factor
and transformed `capture_tokens`). `sampling_graph` retains source and target
graph coordinates separately from the eager measurement contract.
Coverage records completion, completed rounds and the table checksum.
`contract.expert_conditioning` records the policy
(`single-pass-rotated-weight-cycles`), the L2 capacity, the bank multiple and
the warmup cycle count. Tables measured earlier with a lower-bank control pass,
policy `rotated-weight-cycles`, additionally carry a `conditioning_summary`
(`checked_points`, `failed_points`, `max_relative_change`) and a tolerance, and
only those with no failed point were published. Serving reads both forms and
checks each against its own record; nothing acquires the controlled form any
more.
`samples.jsonl` preserves individual forwards for resume and auditing; runtime
lookup needs only root metadata and the component CSV.
Acquisition failures may additionally retain `failures.jsonl` and a diagnostic
Chrome trace for a failed attribution check. These are optional debugging
artifacts emitted on failure, not normal-path metadata or simulator inputs; do
not commit diagnostic artifacts as part of a published profile bundle.
New acquisitions do not write standalone index, contract or coverage JSON files.
Historical standalone layouts remain readable when their method contracts are
current; file-layout fallback does not translate obsolete symbolic names.
Moving metadata does not alter timing-table bytes or turn them
into fresh measurements. Paths locate tables, not execution identity.

Re-running the same contract resumes completed rounds. An incomplete trailing raw
record is archived before repair; malformed complete records are rejected. A
target has an exclusive `writer.lock` and immutable semantic conditions. `--force`
is rejected: use another output root for a fresh acquisition. Run only one
acquisition writer per target; distinct targets serialize their shared root
metadata updates through `.meta.lock`. Changed version/provenance observations
alone do not change resume identity. Raw samples are needed to resume acquisition,
but not to consume a published metadata-and-CSV bundle.
No-op resume preserves the original runtime observations. Distinct later
observations enter the root entry's `acquisition_contexts` only after validated
new work; this is a context list, not a count of durable or completed rows.
Incomplete or quality-failing acquisitions retain raw evidence but do not publish
a new runtime entry. A historical standalone index is consulted only when the root
MoE owner is absent; a malformed or incomplete root entry cannot fall back to it.

Serving automatically selects a matching index entry. Local components use
one-dimensional interpolation. New expert tables build an affine triangulation
of measured N/A/M coordinates once at load time, removing dependent dimensions
such as EP1's `M=N*k`. Lookup uses at most four measured vertices with nonnegative
weights; exact rows are returned directly and queries are cached with bounded
capacity. There is no online nearest-neighbour search, fitted predictor or
per-workload multiplier. Out-of-support queries do not extrapolate.

The expert query is `(gathered/padded N, rank-local A, rank-local M)`, and all
three coordinates enter the lookup cache key. Local gate/finalization components
use their own row counts rather than expert assignments. The router supplies
active counts and local assignments. BALANCED/CUSTOM retain
rounded expected rank loads; RR/RAND count their selected pairs, and CAPTURED
the captured pairs of the scheduled tokens. CUSTOM's active
count curve does not provide a rank-load histogram. Historical balanced-total
component tables keep their old interpolation and mode choice, and support only
their previously admitted routing policies. Retained `moe.csv` reads are unchanged.

Missing deployment coverage or an unsupported routing policy retains the legacy
EP-table path only when a historical whole-block table exists, with its existing
warning and unverified DP/backend accuracy. A fresh bundle without that table
requires a complete matching component acquisition. Corrupt contracts,
checksum mismatches and queries outside an installed table's supported geometry
are errors rather than silently extrapolated measurements.
Deployment selection caches both loaded tables and absent coverage for the run,
sharing these caches through the persistent profile bundle rather than recreating
them in each batch view. Table validation and interpolation-surface construction
therefore occur once per deployment, keyed by TP, DP, rank and sequence-parallel
mode. Corrupt contracts are not cached as missing coverage. Restart the simulation
after changing a bundle; cache reuse does not change lookup or routing arithmetic.

## Runtime communication and limits

The supported DP+EP modular path emits gate/routing, three logical AllGathers (hidden
states, top-k weights, top-k IDs), experts, ReduceScatter, local finalization and,
when the table measures one, the shared expert, in dependency order. Tensor widths
come from recorded dtypes. For TP greater than one, the audited wrapper also
restores TP output with AllGather for sequence parallelism or AllReduce otherwise.
Native components and collectives are charged separately; collective time is not
included in the component CSV.
Modular DP1 emits gate/routing, experts and any shared expert, followed by TP
AllReduce only when TP>1.
There are no DP AllGathers, ReduceScatter or finalize-copy in that trace.
The live TP1 path emits gate/routing, experts and wrapper without collectives;
wrapper work can surround the expert call in the native forward, so this order
is a serial duration representation, not a claim about concurrent execution.

The analytical path retains a worst-rank Ring envelope for unequal DP counts.
Logical tensors and dependency order do not reproduce NCCL's grouped launch,
channel selection or arbitrary ragged task schedule. Optional
[collective links](../reference/cluster-config#collective-specific-links) change
effective communication parameters without changing these tensor shapes.

The modular adapter requires positive DP, full EP, linear expert placement and
native modular experts with standard activation layout: unquantized, or vLLM FP8
with block scales and dynamic activations at DP1. Its unquantized local and
DP+EP contracts remain unchanged. The live adapter requires TP1/DP1/EP1 and
standard token layout; monolithic kernels must expose verifiable native routing
replay. Quantization or shared experts alone are not rejection conditions, but
an unsupported API, deferred finalization, input-weighted routing, fused shared
gate, padded hidden dimensions, biased or redundant experts, LoRA, EPLB or
tensor-sharded experts still fails explicitly. No removed whole-block acquisition
is substituted. New tables admit rank-level N/A/M variation but do not resolve
arbitrary within-rank expert histograms or establish trained-routing accuracy.
No coverage of other models, hardware or execution paths should be inferred
solely from a matching file format.
