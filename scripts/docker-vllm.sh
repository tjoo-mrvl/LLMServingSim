#!/bin/bash

# Launch vLLM Docker for profiler / bench / validate.
#
# Mounts the LLMServingSim repo root as /workspace so the profiler,
# bench, datasets generators, and shared model configs are all visible:
#
#     /workspace/profiler/            profiler package + scripts
#     /workspace/bench/               bench + validate
#     /workspace/workloads/            workload JSONLs and generators
#     /workspace/configs/model/       HF model configs
#
# The working directory defaults to /workspace so any of the modules
# can be run via ``python -m profiler``, ``python -m bench``, etc.
#
# The official vllm/vllm-openai image already provides vllm, pydantic,
# pyyaml, rich, and huggingface_hub — no extra pip installs required.
#
# vLLM version: v0.20.2 is the active pin. v0.20.0 was the first
# release with DeepSeek-V4 (``deepseek_v4.py`` + ``deepseek_v4_attention.py``)
# in ``vllm/model_executor/models/``; v0.20.2 has the bug-fix cycles
# since. If you change this, also re-validate
# ``profiler/core/hooks/moe_hook.py`` — the MoE forced-routing hook
# patches a method on ``FusedMoE`` whose name has changed across
# versions (``forward_native`` on v0.19.x, ``forward`` on v0.20.x; the
# hook auto-detects, but a future refactor of the router internals may
# break the deeper ``select_experts``/``_compute_routing`` patches).

set -euo pipefail

# Resolve the repo root regardless of where this script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../scripts
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                    # .../LLMServingSim

docker run --name vllm_docker \
  --gpus all \
  -it \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$REPO_ROOT":/workspace \
  --volume "$HOME/.cache/huggingface":/root/.cache/huggingface \
  --shm-size=16g \
  -w /workspace \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.20.2 \
  -c "pip install datasets matplotlib && exec bash"
