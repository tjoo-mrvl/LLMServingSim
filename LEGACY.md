# Legacy mode: DeepSeek-R1 on vLLM 0.20.2 (deprecated)

This branch freezes the DeepSeek-R1 work as it stood before LLMServingSim v1.3.0
was merged into `feat/deepseek_v3`. Nothing here is developed further; it exists
so the vLLM 0.20.2 results stay reproducible with the code that produced them.

| Part | What is frozen |
|---|---|
| Simulator | `serving/` at `b1f06ad` (tag `profiler-v0.20.2-r1`) |
| ASTRA-Sim / Chakra | submodule `astra-sim` at `f82fb3d`, Chakra at `bf11884` |
| Profile | `profiler/perf/H200/deepseek-ai/DeepSeek-R1/bf16/`, measured on H200 with vLLM 0.20.2 by the profiler on this branch |
| Ground truth | 14 vLLM 0.20.2 runs in `bench/results/r1-*` (committed here although `bench/results/` is gitignored) |
| Analysis | `docs/docs/correlation/deepseek-r1-correlation-diagnosis.md` |

The profile is read only by this simulator. The v1.3 simulator uses a different
catalog form, layer names and skew-fit format, and this data is deliberately not
converted to it: current R1 work on `feat/deepseek_v3` uses a new H200 profile
taken with the v1.3 profiler on vLLM 0.31.

## Setup

```bash
git worktree add ../LLMServingSim-legacy legacy/vllm-0.20.2
cd ../LLMServingSim-legacy
git submodule update --init --recursive      # astra-sim f82fb3d, Chakra bf11884
python3.10 -m venv /opt/venv/llmservingsim-legacy
source /opt/venv/llmservingsim-legacy/bin/activate
pip install pyyaml pyinstrument rich msgspec pandas==1.5.3 numpy==1.23.5 matplotlib==3.5.3
./scripts/compile.sh                         # installs Chakra into the venv, builds ASTRA-Sim
pip install protobuf==7.35.1                 # see below
```

Chakra declares `protobuf==6.*`, but its build generates `et_def_pb2.py` with
the newest protoc, and protobuf refuses a runtime older than the generated code.
Install the version named in the `Protobuf Python Version` header of the
installed `et_def_pb2.py` (7.35.1 when this was written); pip then warns about
Chakra's declared range, which is harmless. Keep the venv's `bin` first on
`PATH`: the simulator starts the Chakra converter as `python -m ...`.

## Run

From the repository root, with `--max-num-seqs 256 --max-num-batched-tokens 4096`:

| Run | `--cluster-config` | `--dataset` | `--num-reqs` |
|---|---|---|---|
| EP8 fixed b1 / b16 / b64 | `configs/cluster/deepseek_r1_single_instance.json` | `workloads/fixed-8192-4096-b{1,16,64}.jsonl` | 9 / 24 / 72 |
| EP-off fixed b1 / b16 / b64 | `configs/cluster/deepseek_r1_tp8_ep1_single_instance.json` | same | 9 / 24 / 72 |
| EP8 ShareGPT | `configs/cluster/deepseek_r1_single_instance.json` | `workloads/sharegpt-deepseek-r1-300-sps10.jsonl` | 300 |
| EP-off ShareGPT | `configs/cluster/deepseek_r1_tp8_ep1_single_instance.json` | same | 300 |

```bash
python -m serving --cluster-config configs/cluster/deepseek_r1_single_instance.json \
  --dataset workloads/fixed-8192-4096-b1.jsonl --num-reqs 9 \
  --max-num-seqs 256 --max-num-batched-tokens 4096 --output outputs/r1-fixed-b1.csv
```

Each fixed workload holds eight 1024/64 warm-up requests at t=0 and the measured
8192/4096 batch at t=15 s. Score only the measured requests. This branch's
`bench validate` takes bench TTFT from vLLM's `queued_ts`; the v1.3 scorer takes
it from the request's arrival, which reads 25 ms (EP8) and 39 ms (EP-off) higher
on ShareGPT.

## Reproduced results

Medians in ms, re-run on 2026-10-21 with the setup above, against the correlation
doc (§16, §18). Bench medians use this branch's scorer. Every run matches the doc
to its rounding except EP8 b1 TTFT, 0.5 ms (0.2%) above the doc's 255.2. The
doc's runs used an earlier ASTRA-Sim build made in the sim container (May 2026);
these used a fresh build of the pinned commit. The difference was not investigated.

| Run | Bench TTFT / TPOT | Simulator TTFT / TPOT | Doc TTFT / TPOT |
|---|---|---|---|
| EP8 fixed b1 | 388.7 / 9.82 | 255.7 / 14.30 | 255 / 14.3 |
| EP8 fixed b16 | 3505.4 / 20.09 | 2574.5 / 26.22 | 2574 / 26.2 |
| EP8 fixed b64 | 13208.4 / 39.77 | 9654.3 / 46.19 | 9654 / 46.2 |
| EP8 ShareGPT | 76.3 / 40.06 | 58.3 / 37.99 | 58.3 / 38.0 |
| EP-off fixed b1 | 538.1 / 12.75 | 265.1 / 14.52 | 265 / 14.5 |
| EP-off fixed b16 | 4833.7 / 22.77 | 2654.0 / 26.95 | 2654 / 27.0 |
| EP-off fixed b64 | 18106.2 / 48.53 | 9950.4 / 46.69 | 9950 / 46.7 |
| EP-off ShareGPT | 117.7 / 59.32 | 62.1 / 39.06 | 62.1 / 39.1 |

## Known limits

These are properties of this simulator and profile, recorded here, not fixed here.

- TP1 x PP8 (`deepseek_r1_tp1_pp8_single_instance.json`) never produces a token: the
  run stalls from its first batch.
- The dense MLP rows (`gate_up_proj`, `act_fn`, `down_proj`) average the 18432-wide
  dense MLP with the 2048-wide shared expert: the catalog has no
  `not_within: DeepseekV2MoE`.
- `kv_b_proj` was measured only at 528 tokens and above, and its extrapolation is
  charged on decode steps, where vLLM runs the weight-absorbed path without it:
  0.62 ms per decode step at TP8.
- A step that holds two prefill requests prices attention as one chunk over their
  summed context. At TP8 a request's last 2 tokens (8,190 tokens of context) plus
  the next request's first 4,093 are charged 47.7 ms of attention, against 12.3 ms
  for a fresh 4,096-token chunk, which is about the work those two pieces do.
- At TP8 / DP1 the MoE block is charged an all-gather plus a reduce-scatter; vLLM
  0.20.2 uses all-to-all kernels only when `dp_size > 1` and otherwise all-reduces
  the MoE output.
- `skew.csv` timings are 3x too large (summed over the 3 measurement iterations);
  the skew fit uses their ratio, so results are unaffected.
- The workloads of `r1-fixed-b256` (22 MB) and the two SWE-bench runs (84 MB) are
  not included.
