import os, threading
from .utils import get_config
from .radix_tree import *
import logging
from enum import Enum

GB_TO_BYTE = 1024 * 1024 * 1024
MB_TO_BYTE = 1024 * 1024
KB_TO_BYTE = 1024

class Device(Enum):
    NPU = 1
    CPU = 2
    CXL = 3


# Bytes per element by weight-quantization scheme. Keys match vLLM's
# ``--quantization`` choices and HF ``quantization_config.quant_method`` values.
_QUANT_BYTES = {
    'fp8': 1, 'fp8_e4m3': 1, 'fp8_e5m2': 1,
    'int8': 1,
    'int4': 0.5, 'awq': 0.5, 'gptq': 0.5,
    # FP4 schemes (DeepSeek-V4 routed experts use ``expert_dtype: fp4``).
    'fp4': 0.5, 'mxfp4': 0.5, 'nvfp4': 0.5,
}


def _resolve_weight_fp(act_fp, quantization, config=None):
    """Pick bytes/element for the weight tensors of parameterised layers.

    CLI ``--quantization`` wins; otherwise auto-detect from HF
    ``quantization_config.quant_method``. Unknown / unset → no quantization
    (weights kept at activation dtype). Block-scale overhead (e.g. bf16 scale
    per ``weight_block_size`` tile) is ignored — for [128, 128] it adds
    ~0.012% which is well below other sources of modelling error.
    """
    method = quantization
    if method is None and isinstance(config, dict):
        qcfg = config.get('quantization_config') or {}
        if isinstance(qcfg, dict):
            method = qcfg.get('quant_method')
    if method in (None, 'none', 'auto'):
        return act_fp
    if method in _QUANT_BYTES:
        return _QUANT_BYTES[method]
    # Unknown scheme — fall back to activation dtype rather than guessing.
    return act_fp


def _resolve_moe_weight_fp(act_fp, weight_fp, config=None):
    """Pick bytes/element for routed-MoE-expert weights specifically.

    Some checkpoints (DeepSeek-V4) quantize routed experts more aggressively
    than the rest of the model — FP4 experts on top of FP8 dense weights.
    The HF config exposes this via a top-level ``expert_dtype`` field
    distinct from ``quantization_config.quant_method``.

    Falls back to ``weight_fp`` when ``expert_dtype`` is absent — preserves
    the existing single-precision behaviour for Mixtral / Qwen3-MoE /
    phi-mini-MoE / DeepSeek-R1, which all leave routed experts at the same
    precision as the dense weights.
    """
    if isinstance(config, dict):
        ed = config.get('expert_dtype')
        if isinstance(ed, str) and ed in _QUANT_BYTES:
            return _QUANT_BYTES[ed]
    return weight_fp

class MemoryModel():
    def __init__(self, model, instance_id, node_id, num_npus, tp_size, npu_mem, cpu_mem, block_size, fp, enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage, cxl_mem=0, ep_size=1, pp_size=1, kv_cache_dtype='auto', quantization=None):
        self.model = model
        self.node_id = node_id
        self.instance_id = instance_id
        self.num_npus = num_npus
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.ep_size = ep_size
        self.npu_mem = npu_mem * GB_TO_BYTE # GB -> Byte
        self.cpu_mem = cpu_mem * GB_TO_BYTE # GB -> Byte
        self.cxl_mem = cxl_mem * GB_TO_BYTE
        self.block_size = block_size
        # Activation / compute dtype in bytes (e.g. bf16 -> 2). `self.fp` kept
        # for back-compat; downstream code that wants bytes-per-activation can
        # use either name.
        self.act_fp = fp // 8
        self.fp = self.act_fp
        self.kv_fp = 1 if kv_cache_dtype == 'fp8' else self.act_fp  # KV cache bytes per element
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_prefix_sharing = enable_prefix_sharing
        self.prefix_storage = prefix_storage

        self.config = get_config(model)
        # Weight quantization: distinct from activation dtype. ``--quantization
        # fp8`` (or HF config ``quantization_config.quant_method``) means GEMM
        # weights are stored at 1 byte/element while activations stay at
        # ``act_fp``. DeepSeek-R1 ships in this W8A16 fp8 scheme by default.
        self.weight_fp = _resolve_weight_fp(self.act_fp, quantization, self.config)
        # MoE expert weights may be quantized more aggressively than dense
        # weights — e.g. DeepSeek-V4 ships fp8 dense + fp4 experts via the
        # top-level ``expert_dtype`` field. Falls back to ``weight_fp`` for
        # every other model.
        self.moe_weight_fp = _resolve_moe_weight_fp(self.act_fp, self.weight_fp, self.config)
        self.n_embd = self.config['hidden_size']
        self.n_layer = self.config['num_hidden_layers']
        self.n_head = self.config['num_attention_heads']
        self.head_dim = self.config.get('head_dim', self.n_embd // self.n_head)
        self.kv_head = self.config.get("num_key_value_heads", self.n_head)  # fallback to n_head if not defined
        self.q_dim = self.n_head * self.head_dim       # total Q projection output dim
        self.kv_dim = self.kv_head * self.head_dim     # total KV projection output dim
        self.vocab_size = self.config['vocab_size']
        # Accept either the Mistral-style ``num_local_experts`` or the
        # HF/Qwen-style ``num_experts`` key — profiler configs track
        # upstream HF naming which varies per family.
        self.is_moe = 'num_local_experts' in self.config or 'num_experts' in self.config or 'n_routed_experts' in self.config
        # MLA (DeepSeek V2/V3): KV cache stores the compressed latent
        # (kv_lora_rank + qk_rope_head_dim) per token instead of full
        # num_kv_heads*head_dim. Detect via the presence of kv_lora_rank
        # and cache the latent footprint for get_kv.
        self.is_mla = 'kv_lora_rank' in self.config
        if self.is_mla:
            self.kv_lora_rank = int(self.config['kv_lora_rank'])
            self.qk_rope_head_dim = int(self.config.get('qk_rope_head_dim', 0))
            self.mla_latent_dim = self.kv_lora_rank + self.qk_rope_head_dim
        # DeepSeek-V4: MLA-derived but with shape changes (kv_lora_rank
        # collapsed into head_dim, no separate kv_b_proj) plus a per-layer
        # sparse-attention indexer overlay and sliding-window attention.
        # Detection: V4 carries the ``compress_ratios`` array (one entry
        # per layer; value 4 instantiates the indexer for that layer,
        # value 128 skips it). Mutually exclusive with V3 ``is_mla``:
        # V3 has kv_lora_rank but no compress_ratios; V4 has
        # compress_ratios but no kv_lora_rank.
        self.is_v4_mla = 'compress_ratios' in self.config
        self.sliding_window = int(self.config.get('sliding_window', 0) or 0)
        self.compress_ratios = list(self.config.get('compress_ratios') or [])
        if self.is_v4_mla:
            self.o_lora_rank = int(self.config['o_lora_rank'])
            self.o_groups = int(self.config['o_groups'])
            self.index_n_heads = int(self.config['index_n_heads'])
            self.index_head_dim = int(self.config['index_head_dim'])
            # Count of layers that instantiate a DeepseekV4Indexer
            # (those with compress_ratios[i] == 4). About half of the
            # 61 layers for V4-Pro, alternating per the config's array.
            self.n_indexed_layers = sum(1 for r in self.compress_ratios if r == 4)
        # DeepSeek-V3 / R1: layers [0, first_k_dense_replace) run plain
        # MLP; the remaining MoE layers run DeepseekV2MoE. Tracked for
        # the per-layer weight aggregation in get_weight().
        self.first_k_dense_replace = int(self.config.get('first_k_dense_replace', 0))

        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)

        # Memory model
        self.weight = self.get_weight() # assume weight is loaded
        self.npu_used = self.weight
        self.cpu_used = 0
        if self.weight > self.npu_mem:
            raise RuntimeError(f"[MemoryModel] [node={self.node_id},inst={self.instance_id}]: Model size {self.weight*self.num_npus//GB_TO_BYTE}GB exceeds total NPU memory {self.npu_mem*self.num_npus//GB_TO_BYTE}GB")

        if enable_prefix_caching:
            one_token_kv_size = self.get_kv(1)
            self.mem_for_kv = self.npu_mem - self.weight
            self.npu_prefix_cache = RadixCache(device='NPU', 
                                               node_id=self.node_id,
                                               instance_id=self.instance_id,
                                               page_size=self.block_size,
                                               capacity=self.mem_for_kv,
                                               kv_size=one_token_kv_size,
                                               enable_kv_cache_events=True,
                                                )
            if prefix_storage is not None:
                if enable_prefix_sharing and prefix_pool is not None:
                    self.second_tier_prefix_cache = prefix_pool
                else:
                    prefix_cache_capacity = 0
                    if prefix_storage == Device.CPU:
                        device = "CPU"
                        prefix_cache_capacity = self.cpu_mem
                    elif prefix_storage == Device.CXL:
                        device = "CXL"
                        prefix_cache_capacity = self.cxl_mem
                    else:
                        raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}]: Device {prefix_storage} is currently not supported as a second tier prefix cache storage")
                    # print("[instance {}] prefix_cache_capacity : {}".format(instance_id, prefix_cache_capacity // GB_TO_BYTE))
                    self.second_tier_prefix_cache = RadixCache(device=device, 
                                                    node_id=self.node_id,
                                                    instance_id=self.instance_id,
                                                    page_size=1,
                                                    capacity=prefix_cache_capacity,
                                                    kv_size=(one_token_kv_size * self.num_npus),
                                                    enable_kv_cache_events=True,
                                                    )
                
        # Hash id -> token length for corresponding prefix cache block
        self._npu_cache_hashtolen = {}
        self._cpu_cache_hashtolen = {}
        self._bytes_per_token = self.get_kv(1)  # bytes per token for kv cache
    def get_weight(self):
        """Per-GPU model weight in bytes.

        Conservative upper bound across PP ranks: assumes a single rank
        holds embedding + final_layernorm + lm_head along with its share
        of transformer blocks (n_layer // pp_size). In real PP these
        non-block weights live on the first/last rank only, so middle
        ranks are lighter — but using the heaviest-rank value here keeps
        the `weight > npu_mem` check safe.
        """
        tp = self.tp_size
        pp = max(self.pp_size, 1)
        ep = self.ep_size
        fp = self.act_fp
        wfp = self.weight_fp
        mwfp = self.moe_weight_fp
        weight = 0

        _, embedding, _ = calculate_sizes(self.model, 'embedding', 1, parallel=tp, fp=fp, weight_fp=wfp)
        weight += embedding
        # Hybrid-FFN models (DeepSeek-V3) run a plain MLP for the first
        # ``first_k_dense_replace`` layers and the MoE block for the rest.
        # is_moe + first_k_dense_replace=0 collapses to "all MoE", which
        # matches non-hybrid MoE families like Mixtral / Qwen3-MoE.
        # Layer counts are divided by ``pp`` (upstream's per-PP-rank estimate;
        # exact for pp=1, which all DeepSeek configs use).
        if self.is_moe and self.first_k_dense_replace > 0:
            n_dense = self.first_k_dense_replace
            n_moe = self.n_layer - n_dense
            weight += self._get_weight_per_block(tp, ep, fp, wfp, mwfp, mlp_type='dense') * (n_dense // pp)
            weight += self._get_weight_per_block(tp, ep, fp, wfp, mwfp, mlp_type='moe') * (n_moe // pp)
        else:
            mlp_type = 'moe' if self.is_moe else 'dense'
            weight += self._get_weight_per_block(tp, ep, fp, wfp, mwfp, mlp_type=mlp_type) * (self.n_layer // pp)
        _, ln_f, _ = calculate_sizes(self.model, 'final_layernorm', 1, parallel=tp, fp=fp, weight_fp=wfp)
        weight += ln_f
        _, lm_head, _ = calculate_sizes(self.model, 'lm_head', 1, parallel=tp, fp=fp, weight_fp=wfp)
        weight += lm_head

        # DeepSeek-V4: indexer weights are per-indexed-layer, not per-block,
        # so they're aggregated here rather than inside _get_weight_per_block.
        # Each indexer carries idx_wq_b + idx_weights_proj + idx_k_norm
        # (ReplicatedLinear / LayerNorm — not TP-sharded, identical on every
        # rank). idx_compressor and idx_indexer_op are kernel-only (no
        # learnable weights).
        if self.is_v4_mla and self.n_indexed_layers > 0:
            indexer_w = 0
            for ln in ('idx_wq_b', 'idx_weights_proj', 'idx_k_norm'):
                _, w, _ = calculate_sizes(self.model, ln, 1, parallel=tp, fp=fp, weight_fp=wfp)
                indexer_w += w
            weight += indexer_w * self.n_indexed_layers

        self.logger.info(
            "NPU: model weight %dMB loaded",
            weight * tp // MB_TO_BYTE,
        )
        return weight

    def _get_weight_per_block(self, tp, ep, fp, wfp, mwfp, mlp_type):
        """Per-block weight: dense layers use TP, MoE experts use EP.

        ``mlp_type`` selects whether this layer's MLP is the plain dense
        FFN or the MoE block — needed for hybrid models (DeepSeek-V3)
        where layer index decides.
        ``wfp`` is the weight-quantized bytes/element used for GEMM weights;
        ``mwfp`` is the (possibly more aggressive) bytes/element used for
        routed-MoE-expert weights — falls back to ``wfp`` when the model
        doesn't declare a separate ``expert_dtype``. Layernorms keep ``fp``
        (the activation dtype) since RMSNorm scales are not quantized.
        """
        block_weight = 0
        _, ln_w, _ = calculate_sizes(self.model, 'layernorm', 1, parallel=tp, fp=fp, weight_fp=wfp)
        block_weight += ln_w  # input layernorm
        if self.is_v4_mla:
            # DeepSeek-V4 MLA path: slimmer fused A-projection (no
            # separate kv_b_proj — KV stays in latent until the kernel),
            # plus a grouped low-rank o-projection split across
            # ``wo_a`` (ColumnParallel + bmm across o_groups) and
            # ``wo_b`` (RowParallel) through the o_lora_rank bottleneck.
            # Indexer submodules are per-layer-conditional and summed
            # separately in get_weight, not here.
            for ln in ('fused_wqa_wkv', 'q_norm', 'wq_b', 'kv_norm', 'wo_a', 'wo_b'):
                _, w, _ = calculate_sizes(self.model, ln, 1, parallel=tp, fp=fp, weight_fp=wfp)
                block_weight += w
        elif self.is_mla:
            # MLA attention path: fused down-proj + Q/KV norms + B-projections + o_proj.
            # Standard ``qkv_proj`` is replaced by the small fused A-projection
            # and the two B-projections (Q and KV up-projections).
            for ln in ('fused_qkv_a_proj', 'q_a_layernorm', 'q_b_proj',
                       'kv_a_layernorm', 'kv_b_proj', 'o_proj'):
                _, w, _ = calculate_sizes(self.model, ln, 1, parallel=tp, fp=fp, weight_fp=wfp)
                block_weight += w
        else:
            _, qkv_w, _ = calculate_sizes(self.model, 'qkv_proj', 1, parallel=tp, fp=fp, weight_fp=wfp)
            block_weight += qkv_w
            _, o_w, _ = calculate_sizes(self.model, 'o_proj', 1, parallel=tp, fp=fp, weight_fp=wfp)
            block_weight += o_w
        block_weight += ln_w  # post layernorm (same weight size)
        if mlp_type == 'moe':
            # EP distributes whole experts (parallel=ep); when tp>ep (EP-off /
            # partial-EP) each expert is ALSO TP-sharded along its intermediate
            # dim by moe_tp = tp//ep. ep==tp -> moe_tp=1 (bit-identical).
            _, moe_w, _ = calculate_sizes(self.model, 'moe', 1, parallel=ep, fp=fp,
                                          weight_fp=wfp, moe_weight_fp=mwfp,
                                          moe_tp=max(1, tp // ep))
            block_weight += moe_w
        else:
            _, ffn1_w, _ = calculate_sizes(self.model, 'gate_up_proj', 1, parallel=tp, fp=fp, weight_fp=wfp)
            block_weight += ffn1_w
            _, ffn2_w, _ = calculate_sizes(self.model, 'down_proj', 1, parallel=tp, fp=fp, weight_fp=wfp)
            block_weight += ffn2_w
        return block_weight

    def get_kv(self, seq):
        # shape of kv cache
        # (kv_head, batch_size, n_embd//n_head, seq_len) per layer
        # return batch_size = 1 to caclulate max batch_size in scheduler

        if self.is_v4_mla:
            # DeepSeek-V4: MLA latent of width head_dim per token, bounded
            # by sliding_window. Replicated across TP ranks (no //num_npus).
            # Indexer layers (compress_ratios[i]==4) carry an additional
            # K-cache for the SparseAttnIndexer — first-order approximated
            # as index_n_heads*index_head_dim compressed keys per token.
            # The exact DeepseekV4IndexerCache layout may include a small
            # constant factor (compressed-vs-original key storage); leave
            # as a follow-up once a real v0.20.2 load can be inspected.
            effective_seq = min(seq, self.sliding_window) if self.sliding_window else seq
            mla_bytes = self.head_dim * effective_seq * self.n_layer * self.kv_fp
            indexer_bytes = (self.index_n_heads * self.index_head_dim
                             * effective_seq * self.n_indexed_layers * self.kv_fp)
            return mla_bytes + indexer_bytes
        if self.is_mla:
            # MLA caches a single per-token latent (kv_lora_rank +
            # qk_rope_head_dim) shared by every head — vLLM stores it
            # pre-``kv_b_proj`` and replicates it across TP ranks, so
            # there's no division by num_npus and no factor of 2.
            return self.mla_latent_dim * seq * self.n_layer * self.kv_fp
        # K & V multiply 2
        return 2 * self.kv_dim * seq * self.n_layer * self.kv_fp // self.num_npus
    
    # get the total size of current kv cache for the request
    # used when adding prefilled request to decode instance.
    def get_total_kv(self, req):
        # ceil division: (n + block_size - 1) // block_size
        num_blocks = (req.num_computed_tokens + self.block_size - 1) // self.block_size
        return self.get_kv(num_blocks * self.block_size)

    # get size of kv block that should be 'added'. including new init requests
    # also checks evicted request and include its kv cache
    # scheduled_tokens: dict mapping request id to number of tokens scheduled this step
    # 
    # vLLM-style cumulative allocation:
    #   blocks_after = ceil((computed + scheduled) / block_size)
    #   blocks_before = ceil(computed / block_size) if computed > 0 else 0
    #   new_blocks = blocks_after - blocks_before
    def get_block_kv(self, batch_req, batch_len, scheduled_tokens=None):
        # print("[get_block_kv] current batch_req length : {}".format(batch_len))
        block_kv_size = 0
        for i in range(batch_len):
            req = batch_req[i]
            if req.evict or req.is_prefill():
                # Prefill and reloaded decode requests may allocate newly
                # computed blocks. Existing evicted KV is reloaded separately
                # by Scheduler.load_size.
                hit = req.npu_cache_hit if self.enable_prefix_caching else 0
                
                if scheduled_tokens and req.id in scheduled_tokens:
                    tokens_this_step = scheduled_tokens[req.id]
                else:
                    raise RuntimeError("[MemoryModel] [node_id={self.node_id},inst={self.instance_id}]: scheduled_tokens cannot be None")
                
                # vLLM-style cumulative block allocation
                computed_before = req.num_computed_tokens
                
                total_after = computed_before + tokens_this_step
                
                # Calculate blocks needed (cumulative)
                blocks_after = (total_after + self.block_size - 1) // self.block_size
                blocks_before = (computed_before + self.block_size - 1) // self.block_size if computed_before > 0 else 0
                
                
                new_blocks = max(0, blocks_after - blocks_before)
                block_kv_size += self.get_kv(new_blocks * self.block_size)
                # print("[DEBUG] hit : {} | tokens_this_step : {} | computed_before : {} | total_after : {} | new_blocks : {} | block_kv_size : {}".format(
                #     hit, tokens_this_step, computed_before, total_after, new_blocks, block_kv_size
                # ))
            else:
                # Decode: use num_computed_tokens (or input for backwards compat)
                computed = req.num_computed_tokens
                num_before = (computed + self.block_size - 1) // self.block_size if computed > 0 else 0
                num_after = (computed + 1 + self.block_size - 1) // self.block_size
                if num_after > num_before: # difference of the block is maximum one block
                    block_kv_size += self.get_kv(self.block_size)
        return block_kv_size
    
    # get size of kv cache that should be evicted
    def get_evict_kv(self, req):
        evict_size = 0
        # Use num_computed_tokens if available, fallback to input for backwards compat
        computed = req.num_computed_tokens
        hit = req.npu_cache_hit if self.enable_prefix_caching else 0
        needed = max(0, computed - hit)
        # ceil division: (needed + block_size - 1) // block_size
        num_blocks = (needed + self.block_size - 1) // self.block_size
        evict_size += self.get_kv(num_blocks * self.block_size)
        return evict_size

    def free_weight(self):
        if self.npu_used - self.weight < 0:
            raise RuntimeError(
                f"[MemoryModel] [node={self.node_id}, inst={self.instance_id}] NPU: tried to free model weight {self.weight / MB_TO_BYTE:.2f}MB "
                f"but only {self.npu_used / MB_TO_BYTE:.2f}MB is used."
            )
        self.logger.info(
            "NPU: used: %.2fMB remove: %.2fMB after: %.2fMB",
            self.npu_used / MB_TO_BYTE,
            self.weight / MB_TO_BYTE,
            (self.npu_used - self.weight) / MB_TO_BYTE,
        )
        self.npu_used -= self.weight

    def is_free(self):
        is_free = self.npu_used == 0 and self.cpu_used == 0
        if not is_free:
            self.logger.error(
                "Memory leak detected: NPU used: %.2fMB, CPU used: %.2fMB",
                self.node_id,
                self.instance_id,
                self.npu_used / MB_TO_BYTE,
                self.cpu_used / MB_TO_BYTE,
            )
        return

    # -------------------- Memory Management --------------------
    
    def allocate(self, size, device):
        if device == Device.NPU:
            if self.npu_used + size > self.npu_mem:
                raise RuntimeError(
                    f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] NPU: tried to load {size / MB_TO_BYTE:.2f}MB but only {(self.npu_mem - self.npu_used) / MB_TO_BYTE:.2f}MB is available."
                )
            self.logger.info(
                "NPU: used: %.2fMB load: %.2fMB after: %.2fMB",
                self.npu_used / MB_TO_BYTE,
                size / MB_TO_BYTE,
                (self.npu_used + size) / MB_TO_BYTE,
            )
            self.npu_used += size
        elif device == Device.CPU:
            if self.prefix_storage == Device.CPU and self.enable_prefix_sharing:
                self.second_tier_prefix_cache.allocate(size)
            else:
                if self.cpu_used + size > self.cpu_mem:
                    raise RuntimeError(
                        f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] CPU: tried to load {size / MB_TO_BYTE:.2f}MB "
                        f"but only {(self.cpu_mem - self.cpu_used) / MB_TO_BYTE:.2f}MB is available."
                    )
                self.logger.info(
                    "CPU: used: %.2fMB load: %.2fMB after: %.2fMB",
                    self.cpu_used / MB_TO_BYTE,
                    size / MB_TO_BYTE,
                    (self.cpu_used + size) / MB_TO_BYTE,
                )
                self.cpu_used += size
        elif device == Device.CXL:
            self.second_tier_prefix_cache.allocate(size)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to allocate KV cache in unsupported device {device}")
    
    def free(self, size, device):
        if device == Device.NPU:
            if self.npu_used - size < self.weight:
                raise RuntimeError(
                    f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] NPU: tried to free {size / MB_TO_BYTE:.2f}MB but only {(self.npu_used - self.weight) / MB_TO_BYTE:.2f}MB is used."
                )
            self.logger.info(
                "NPU: used: %.2fMB remove: %.2fMB after: %.2fMB",
                self.npu_used / MB_TO_BYTE,
                size / MB_TO_BYTE,
                (self.npu_used - size) / MB_TO_BYTE,
            )
            self.npu_used -= size

        elif device == Device.CPU:
            if self.prefix_storage == Device.CPU and self.enable_prefix_sharing:
                self.second_tier_prefix_cache.free(size)
            else:
                if self.cpu_used - size < 0:
                    raise RuntimeError(
                        f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] CPU: tried to free {size / MB_TO_BYTE:.2f}MB "
                        f"but only {self.cpu_used / MB_TO_BYTE:.2f}MB is used."
                    )
                self.logger.info(
                    "CPU: used: %.2fMB remove: %.2fMB after: %.2fMB",
                    self.cpu_used / MB_TO_BYTE,
                    size / MB_TO_BYTE,
                    (self.cpu_used - size) / MB_TO_BYTE,
                )
                self.cpu_used -= size
        elif device == Device.CXL:
            self.second_tier_prefix_cache.free(size)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to free KV cache in unsupported device {device}")
    
    def is_avail(self, size, device):
        if device == Device.NPU:
            if self.npu_mem - self.npu_used >= size:
                return True
            else:
                return False 
        elif device == Device.CPU:
            if self.enable_prefix_sharing:
                return self.second_tier_prefix_cache.is_avail(size)
            else:
                if self.cpu_mem - self.cpu_used >= size:
                    return True
                else:
                    return False 
        elif device == Device.CXL:
            return self.second_tier_prefix_cache.is_avail(size)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to check available size of unsupported device {device}")
    
    def need_size(self, size, device):
        if device == Device.NPU:
            needed = (size - (self.npu_mem - self.npu_used))
            if needed > 0:
                return needed
            else:
                return 0
        elif device == Device.CPU:
            if self.enable_prefix_sharing:
                return self.second_tier_prefix_cache.need_size(size)
            else:
                needed = (size - (self.cpu_mem - self.cpu_used))
                if needed > 0:
                    return needed
                else:
                    return 0
        elif device == Device.CXL:
            return self.second_tier_prefix_cache.need_size(size)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to check available size of unsupported device {device}")

    def avail_size(self, device):
        if not self.enable_prefix_caching:
            return 0
        
        if device == Device.NPU:
            return self.npu_prefix_cache.avail_size()
        elif device == Device.CPU or device == Device.CXL:
            return self.second_tier_prefix_cache.avail_size()
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to get available size of prefix cache in unsupported device {device}")
    
    # -------------------- Prefix Cache Management --------------------

    def storage_cache_evicted_req(self, req):
        if self.enable_prefix_caching:
            new_last_node = self.second_tier_prefix_cache.cache_unfinished_req(req, update=False) # do not update hit counts
            # should lock evicted kv cache in cpu
            self.second_tier_prefix_cache.inc_lock_ref(new_last_node)
            req.cpu_last_node = new_last_node
            self.apply_kv_cache_events()

    def evictable_size(self, device):
        if not self.enable_prefix_caching:
            return 0
        
        if device == Device.NPU:
            return self.npu_prefix_cache.evictable_size() * self._bytes_per_token
        elif device == Device.CPU or device == Device.CXL:
            return self.second_tier_prefix_cache.evictable_size() * self._bytes_per_token
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to get evictable size of prefix cache in unsupported device {device}")


    def lock_prefix(self, req, device): 
        # Increment lock ref count on req.npu_last_node (set by prefix_match)
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU and req.npu_last_node is not None:
            node = req.npu_last_node
            # print(f"[LOCK] req={req.id} lock_prefix node_id={node.id} lock_ref_BEFORE={node.lock_ref}")
            self.npu_prefix_cache.inc_lock_ref(req.npu_last_node)
            # print(f"[LOCK] req={req.id} lock_prefix node_id={node.id} lock_ref_AFTER={node.lock_ref}")
        elif (device == Device.CPU or device == Device.CXL) and req.cpu_last_node is not None:
            self.second_tier_prefix_cache.inc_lock_ref(req.cpu_last_node)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to lock prefix cache in unsupported device {device}")
    
    def unlock_prefix(self, req, device):
        # Decrement lock ref count on req.npu_last_node (set by prefix_match)
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU and req.npu_last_node is not None:
            node = req.npu_last_node
            # print(f"[UNLOCK] req={req.id} unlock_prefix node_id={node.id} lock_ref_BEFORE={node.lock_ref}")
            self.npu_prefix_cache.dec_lock_ref(req.npu_last_node)
            # print(f"[UNLOCK] req={req.id} unlock_prefix node_id={node.id} lock_ref_AFTER={node.lock_ref}")
            req.npu_last_node = None
            req._prefix_locked = False
        elif device == Device.CPU and req.cpu_last_node is not None:
            self.second_tier_prefix_cache.dec_lock_ref(req.cpu_last_node)
            req.cpu_last_node = None
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to unlock prefix cache in unsupported device {device}")
    
    def cache_unfinished_req(self, req, device):
        # Get new_last_node via cache_unfinished_req (replaces last node)
        # Decrement old node's lock ref count, increment new node's lock ref count
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU:
            new_last_node = self.npu_prefix_cache.cache_unfinished_req(req)
            
            old_node = req.npu_last_node
            # print(f"[CACHE_UNFINISHED] req={req.id} old_node_id={old_node.id if old_node else None}(lock_ref={old_node.lock_ref if old_node else 'N/A'}) -> new_node_id={new_last_node.id}(lock_ref={new_last_node.lock_ref})")
            if old_node is not None and req._prefix_locked:
                self.npu_prefix_cache.dec_lock_ref(old_node)
            self.npu_prefix_cache.inc_lock_ref(new_last_node)
            # print(f"[CACHE_UNFINISHED] req={req.id} AFTER: old_node_id={old_node.id}(lock_ref={old_node.lock_ref}) new_node_id={new_last_node.id}(lock_ref={new_last_node.lock_ref})")
            req.npu_last_node = new_last_node
            req._prefix_locked = True
            if self.logger.isEnabledFor(logging.DEBUG):
                # print(f"cache_unfinished_req of req {req.id}")
                # print(f"===============NPU PREFIX CAHCE of Instance[{self.instance_id}]=================")
                self.npu_prefix_cache.pretty_print()
        elif device == Device.CPU or device == Device.CXL:
            self.second_tier_prefix_cache.cache_unfinished_req(req)
            if self.logger.isEnabledFor(logging.DEBUG):
                # print(f"cache_unfinished_req of req {req.id}")
                # print(f"===============AFTER INSERT: {self.second_tier_prefix_cache.device} PREFIX CAHCE at pid={os.getpid()} tid={threading.get_ident()} pool_id={id(self.second_tier_prefix_cache)}, size={self.second_tier_prefix_cache.total_size()}=================")
                self.second_tier_prefix_cache.pretty_print()
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to cache prefix cache of unfinished request to unsupported device {device}")
        
        self.apply_kv_cache_events()

    def cache_finished_req(self, req, device):
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU:
            self.npu_prefix_cache.cache_finished_req(req)
            # Only dec_lock_ref if the request was locked
            node = req.npu_last_node
            if not req._prefix_locked:
                # Never locked → skip dec
                pass
                # print(f"[CACHE_FINISHED] req={req.id} node_id={node.id if node else None} lock_ref={node.lock_ref if node else 'N/A'} (SKIPPED dec - not locked)")
            else:
                # print(f"[CACHE_FINISHED] req={req.id} node_id={node.id if node else None} lock_ref_BEFORE={node.lock_ref if node else 'N/A'}")
                if node is not None:
                    self.npu_prefix_cache.dec_lock_ref(node)
                    req.npu_last_node = None
                req._prefix_locked = False
            # node = req.npu_last_node
            # print(f"[CACHE_FINISHED] req={req.id} node_id={node.id if node else None} lock_ref_BEFORE={node.lock_ref if node else 'N/A'}")
            # self.npu_prefix_cache.dec_lock_ref(req.npu_last_node)
                # print(f"[CACHE_FINISHED] req={req.id} node_id={node.id if node else None} lock_ref_AFTER={node.lock_ref if node else 'N/A'}")
            # print(f"[CACHE_FINISHED] req={req.id} evictable_size={self.npu_prefix_cache.evictable_size()} protected_size={self.npu_prefix_cache.protected_size()} total_size={self.npu_prefix_cache.total_size()}")
            if self.logger.isEnabledFor(logging.DEBUG):
                print(f"cache_finished_req of req {req.id}")
                print(f"===============NPU PREFIX CACHE of Instance[{self.instance_id}]=================")
                self.npu_prefix_cache.pretty_print()
        elif device == Device.CPU or device == Device.CXL:
            self.second_tier_prefix_cache.cache_finished_req(req)
            if self.logger.isEnabledFor(logging.DEBUG):
                # print(f"cache_finished_req of req {req.id}")
                # print(f"===============AFTER INSERT: {self.second_tier_prefix_cache.device} PREFIX CAHCE at pid={os.getpid()} tid={threading.get_ident()} pool_id={id(self.second_tier_prefix_cache)}, size={self.second_tier_prefix_cache.total_size()}=================")
                self.second_tier_prefix_cache.pretty_print()
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to cache prefix cache of finished request to unsupported device {device}")
        
        self.apply_kv_cache_events()

    def evict_prefix_cache(self, bytes, device):
        if not self.enable_prefix_caching or bytes <= 0:
            return

        if device == Device.NPU:
            cache = self.npu_prefix_cache
        elif device == Device.CPU:
            cache = self.second_tier_prefix_cache
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to evict prefix cache to unsupported device {device}")

        # Each cache instance carries its own bytes-per-token in kv_size:
        # per-rank for NPU, full-cluster for the second-tier pool.
        space_needed = (bytes + cache.kv_size - 1) // cache.kv_size
        cache.evict(space_needed)

        self.apply_kv_cache_events()

    # -------------------- Prefix Cache Helpers --------------------

    def prefix_match(self, req): # req.prefix_cache_hit initialization 
        if not self.enable_prefix_caching:
            return
        
        tokens = req.input_hash_ids
        if tokens is None:
            return
        old_node = req.npu_last_node
        res = self.npu_prefix_cache.match_prefix(tokens[:req.input])
        req.npu_cache_hit = res.hit_length
        req.npu_last_node = res.last_device_node
        # print(f"[PREFIX_MATCH] req={req.id} old_node_id={old_node.id if old_node else None}(lock_ref={old_node.lock_ref if old_node else 'N/A'}) -> new_node_id={res.last_device_node.id}(lock_ref={res.last_device_node.lock_ref}) hit={res.hit_length} num_computed={req.num_computed_tokens}")

        if self.prefix_storage is not None:
            res_storage = self.second_tier_prefix_cache.match_prefix(tokens[:req.input])
            req.storage_cache_hit = res_storage.hit_length
            req.storage_last_node = res_storage.last_device_node
        else:
            req.storage_cache_hit = 0
            req.storage_last_node = None
        
        req.prefix_cache_hit = max(req.npu_cache_hit, req.storage_cache_hit)
        # if req.num_computed_tokens < req.prefix_cache_hit:
        #     req.num_computed_tokens = req.prefix_cache_hit
        if req.num_computed_tokens == 0:
            req.num_computed_tokens = req.prefix_cache_hit
            # print(f"Request[{req.id}] prefix cache hit: {req.prefix_cache_hit} tokens (NPU: {req.npu_cache_hit}, {self.prefix_storage}: {req.storage_cache_hit})")
        # for debugging
        
        # print(f"===============NPU PREFIX CAHCE of Instance[{self.instance_id}]=================")
        # self.npu_prefix_cache.pretty_print()
        # print("===============CPU PREFIX CAHCE=================")
        # self.second_tier_prefix_cache.pretty_print()
    
    def erase_prefix_info(self, req):
        if not self.enable_prefix_caching:
            return
        
        req.prefix_cache_hit = 0
        req.npu_cache_hit = 0
        req.storage_cache_hit = 0
        req.npu_last_node = None
        req.storage_last_node = None

    def free_prefix_cache(self):
        if not self.enable_prefix_caching:
            return
        # free evictable prefix cache, if evictable_size != total_size there is locked prefix cache
        self.free(self.npu_prefix_cache.evictable_size() * self._bytes_per_token, Device.NPU)
        if not self.enable_prefix_sharing and self.prefix_storage is not None:
            self.free(self.second_tier_prefix_cache.evictable_size() * self._bytes_per_token * self.num_npus, self.prefix_storage)
    
    # Count load/unload events from prefix cache and update memory usage
    def apply_kv_cache_events(self):
        # if not self.enable_prefix_caching:
        #     return
        npu_byte_alloc = 0
        npu_byte_free = 0
        cpu_byte_alloc = 0
        cpu_byte_free = 0
        # self.npu_prefix_cache.take_events() -> [BlockStored, BlockStored, BlockRemoved, ...]
        for ev in self.npu_prefix_cache.take_events():
            # print(f" current event block: {ev}")
            if isinstance(ev, BlockStored):
                tlen = len(ev.token_ids)
                for h in ev.block_hashes:
                    # self._npu_cache_hashtolen[h] = tlen
                    if h in self._npu_cache_hashtolen:
                        self._npu_cache_hashtolen[h][1] += 1
                        # if self._npu_cache_hashtolen[h][1] >= 2:
                        #     print("duplicated hash occurs!! h : {}".format(h))
                    else:
                        self._npu_cache_hashtolen[h] = [tlen, 1]
                npu_byte_alloc += self.get_kv(tlen)
            elif isinstance(ev, BlockRemoved):
                for h in ev.block_hashes:
                    # tlen = self._npu_cache_hashtolen.pop(h, 0)
                    # if tlen == 0:
                    if h in self._npu_cache_hashtolen:
                        tlen = self._npu_cache_hashtolen[h][0]
                        self._npu_cache_hashtolen[h][1] -= 1
                        if self._npu_cache_hashtolen[h][1] <= 0:
                            del self._npu_cache_hashtolen[h]
                        npu_byte_free += self.get_kv(tlen)
                    else:
                        print(f"[HASH_MISS] BlockRemoved hash={h} NOT FOUND in map (map_size={len(self._npu_cache_hashtolen)})")
                        self.logger.warning(f"NPU prefix cache remove unknown block hash {h}")
                    # else:
                    #     print(f"[HASH_HIT] BlockRemoved hash={h} tlen={tlen}")
                    # npu_byte_free += self.get_kv(tlen)
        # free first, then allocate
        if npu_byte_free > 0:
            self.free(npu_byte_free, Device.NPU)
        if npu_byte_alloc > 0:
            self.allocate(npu_byte_alloc, Device.NPU)
        # if npu_byte_free > 0:
        #     self.free(npu_byte_free, Device.NPU)

        # Second-tier (CPU/CXL) prefix cache events.
        if self.prefix_storage is None:
            return

        if self.prefix_storage is Device.CPU and not self.enable_prefix_sharing:
            # Non-shared CPU second_tier: bridge events into the instance's
            # cpu_used counter so allocations are bounded by cpu_mem.
            for ev in self.second_tier_prefix_cache.take_events():
                if isinstance(ev, BlockStored):
                    tlen = len(ev.token_ids)
                    for h in ev.block_hashes:
                        if h in self._cpu_cache_hashtolen:
                            self._cpu_cache_hashtolen[h][1] += 1
                        else:
                            self._cpu_cache_hashtolen[h] = [tlen, 1]
                    cpu_byte_alloc += self.get_kv(tlen) * self.num_npus
                elif isinstance(ev, BlockRemoved):
                    for h in ev.block_hashes:
                        if h in self._cpu_cache_hashtolen:
                            tlen = self._cpu_cache_hashtolen[h][0]
                            self._cpu_cache_hashtolen[h][1] -= 1
                            if self._cpu_cache_hashtolen[h][1] <= 0:
                                del self._cpu_cache_hashtolen[h]
                            cpu_byte_free += self.get_kv(tlen) * self.num_npus
                        else:
                            self.logger.warning(f"CPU prefix cache remove unknown block hash {h}")

            if cpu_byte_free > 0:
                self.free(cpu_byte_free, Device.CPU)
            if cpu_byte_alloc > 0:
                self.allocate(cpu_byte_alloc, Device.CPU)
        else:
            # Shared pool or CXL: the cache itself accounts via
            # total_memory_usage (= kv_stored + total_size * kv_size),
            # so no instance-side counter update is needed. Drain the
            # queue to prevent it from growing unboundedly.
            self.second_tier_prefix_cache.take_events()

    def return_prefix_info(self):
        if not self.enable_prefix_caching:
            return (0, 0, 0, 0)
        if self.prefix_storage is None:
            return (self.npu_prefix_cache.return_prefix_info(), (0, 0))
        return (self.npu_prefix_cache.return_prefix_info(), self.second_tier_prefix_cache.return_prefix_info())

        
def full_cluster_kv_bytes_per_token(model, fp, kv_cache_dtype='auto'):
    """Bytes of KV cache per token aggregated over the full TP cluster.

    Mirrors MemoryModel.get_kv(1) * num_npus but computes directly, avoiding
    the per-rank floor-division roundoff. ``fp`` is the model weight dtype
    in bits (16, 32, ...). ``kv_cache_dtype='fp8'`` forces 1 byte per element
    for the KV cache regardless of weight dtype.
    """
    config = get_config(model)
    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    head_dim = config.get('head_dim', n_embd // n_head)
    kv_head = config.get('num_key_value_heads', n_head)
    kv_dim = kv_head * head_dim
    n_layer = config['num_hidden_layers']
    kv_fp = 1 if kv_cache_dtype == 'fp8' else fp // 8
    # 2 (K + V) * kv_dim * n_layer * bytes_per_elem
    return 2 * kv_dim * n_layer * kv_fp


# calculate the per-rank input, weight, output size of each layer
def calculate_sizes(model, layer_name, length, kv_len=None, pim=False, parallel=1, fp=2,
                    weight_fp=None, moe_weight_fp=None, moe_tp=1):
    """Calculate input, weight, and output tensor sizes for a given layer.

    Args:
        parallel: parallelism degree for weight/activation sharding.
            For dense layers this is TP; for MoE experts this is EP.
        fp: bytes per activation/compute element (bf16 -> 2, fp8 -> 1).
            Used for input/output sizes, layernorm/embedding/router weights,
            and unquantized parameters.
        weight_fp: bytes per element for quantizable GEMM weights
            (qkv_proj, o_proj, gate_up_proj, down_proj, MLA B-projections,
            lm_head). When ``None`` (default), falls back to ``fp`` —
            i.e. no weight quantization. Set to ``1`` for fp8/int8,
            ``0.5`` for int4/fp4/AWQ/GPTQ. Modelling W8A16 (DeepSeek-R1):
            ``fp=2, weight_fp=1``.
        moe_weight_fp: bytes per element for routed-MoE-expert weights
            specifically. Distinct from ``weight_fp`` because some
            checkpoints (DeepSeek-V4) quantize experts more aggressively
            than the dense path (fp8 dense + fp4 experts). Only consumed
            by the ``moe`` layer branch. When ``None`` (default), falls
            back to ``weight_fp``.
        moe_tp: intra-expert tensor-parallel degree for the ``moe`` branch.
            EP (``parallel``) distributes WHOLE experts across ranks; when
            ``tp > ep`` (EP-off or partial-EP) each expert is ADDITIONALLY
            TP-sharded along its intermediate dim by ``moe_tp = tp // ep``.
            Default ``1`` (ep == tp) leaves the expert weights unsharded —
            bit-identical to the pre-existing expert-parallel model.
    """
    if weight_fp is None:
        weight_fp = fp
    if moe_weight_fp is None:
        moe_weight_fp = weight_fp
    config = get_config(model)
    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    head_dim = config.get('head_dim', n_embd // n_head)
    vocab_size = config['vocab_size']
    kv_head = config.get("num_key_value_heads", n_head)  # fallback to n_head if not defined
    q_dim = n_head * head_dim       # total Q projection output dim
    kv_dim = kv_head * head_dim     # total KV projection output dim
    ffn_dim = config.get("intermediate_size", config.get("ffn_dim"))  # dense FFN dim
    moe_ffn_dim = config.get("moe_intermediate_size", ffn_dim)  # per-expert FFN dim (may differ from dense)
    # Same both-name fallback as MemoryModel.__init__ — HF / Qwen use
    # ``num_experts`` while Mistral uses ``num_local_experts``; DeepSeek
    # uses ``n_routed_experts``.
    num_local_experts = config.get(
        "num_local_experts",
        config.get("num_experts", config.get("n_routed_experts", 1)),
    )
    n_shared_experts = int(config.get("n_shared_experts", 0) or 0)

    # MLA (DeepSeek V2/V3) extras. ``q_total`` is the per-rank Q output
    # of q_b_proj; ``v_total`` is the per-rank attention output that
    # feeds o_proj. ``mla_latent_dim`` is the cached latent width.
    is_mla = 'kv_lora_rank' in config
    if is_mla:
        kv_lora_rank = int(config['kv_lora_rank'])
        q_lora_rank = int(config.get('q_lora_rank', 0)) or None
        qk_nope_head_dim = int(config.get('qk_nope_head_dim', 0))
        qk_rope_head_dim = int(config.get('qk_rope_head_dim', 0))
        v_head_dim = int(config.get('v_head_dim', head_dim))
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        q_total = n_head * qk_head_dim       # full Q output of q_b_proj
        v_total = n_head * v_head_dim        # attention output dim (input to o_proj)
        mla_latent_dim = kv_lora_rank + qk_rope_head_dim
        a_proj_out = (q_lora_rank or 0) + mla_latent_dim
    else:
        kv_lora_rank = q_lora_rank = qk_nope_head_dim = qk_rope_head_dim = 0
        v_head_dim = head_dim
        qk_head_dim = head_dim
        q_total = q_dim
        v_total = q_dim
        mla_latent_dim = 0
        a_proj_out = 0

    # DeepSeek-V4 MLA dims. V4 collapses kv_lora_rank into head_dim and
    # drops the separate kv_b_proj path; the fused A-projection outputs
    # only q_lora_rank + head_dim. The o-projection is decomposed into
    # ``wo_a`` (column-parallel + batched matmul across ``o_groups``)
    # plus ``wo_b`` (row-parallel) through an ``o_lora_rank`` bottleneck.
    is_v4_mla = 'compress_ratios' in config
    if is_v4_mla:
        v4_q_lora_rank = int(config['q_lora_rank'])
        v4_head_dim = int(config['head_dim'])          # = kv_lora_rank in V4
        v4_o_lora_rank = int(config['o_lora_rank'])
        v4_o_groups = int(config['o_groups'])
        v4_a_proj_out = v4_q_lora_rank + v4_head_dim   # fused_wqa_wkv output
        v4_index_n_heads = int(config['index_n_heads'])
        v4_index_head_dim = int(config['index_head_dim'])
        v4_sliding_window = int(config.get('sliding_window', 0) or 0)
        v4_qk_rope_head_dim = int(config.get('qk_rope_head_dim', 0))
    else:
        v4_q_lora_rank = v4_head_dim = v4_o_lora_rank = v4_o_groups = 0
        v4_a_proj_out = v4_index_n_heads = v4_index_head_dim = 0
        v4_sliding_window = v4_qk_rope_head_dim = 0

    p = max(int(parallel), 1)

    # NOTE (vLLM-style assumptions):
    # NOTE (vLLM-style assumptions):
    # - Embedding / LM head: vocab-parallel → split vocab_size across ranks.
    # - Q/K/V: ColumnParallelLinear         → split output dim across ranks.
    # - o_proj: RowParallelLinear           → split input dim across ranks.
    # - LayerNorm weights: replicated (NOT sharded).
    # - MoE experts: parallel = EP degree, each rank holds num_local_experts // p experts.

    # ----------------- Embedding & Norms -----------------
    if layer_name == "embedding":
        # VocabParallelEmbedding is a sparse lookup, not a GEMM; vLLM keeps it
        # at compute dtype even under fp8/int8 quantization.
        input_size = length * fp * 2  # token_ids are int32 or int64
        weight_size = (vocab_size // p) * n_embd * fp
        output_size = length * n_embd * fp

    elif layer_name in ["input_layernorm", "post_layernorm", "final_layernorm", "layernorm"]:
        input_size = length * n_embd * fp
        weight_size = 1 * n_embd * fp  # scale only
        output_size = length * n_embd * fp

    elif layer_name == "qk_norm":
        input_size = length * (q_dim + kv_dim) // p * fp
        weight_size = 2 * head_dim * fp
        output_size = length * (q_dim + kv_dim) // p * fp

    # ----------------- RoPE & Attention Core -----------------
    elif layer_name == "rotary_emb":
        input_size = ((n_head // p) + (kv_head // p)) * length * head_dim * fp
        weight_size = 0
        output_size = ((n_head // p) + (kv_head // p)) * length * head_dim * fp

    elif layer_name == "attention":
        if is_v4_mla:
            # DeepSeek-V4 MLA: Q is (n_head * head_dim) per rank, KV cache
            # is the head_dim-wide latent (== kv_lora_rank in V4) bounded
            # by sliding_window. Latent is replicated across TP ranks (no // p).
            # Indexer K-cache contribution is accounted for in get_kv on
            # the layers where compress_ratios[i]==4; not double-counted here.
            attn_len = length if not pim else 1
            kv_for_attn = kv_len if kv_len is not None else attn_len
            if v4_sliding_window > 0:
                kv_for_attn = min(kv_for_attn, v4_sliding_window)
            input_size = (
                (n_head // p) * attn_len * v4_head_dim * fp +
                v4_head_dim * kv_for_attn * fp        # latent KV, replicated
            )
            weight_size = 0
            output_size = (n_head // p) * attn_len * v4_head_dim * fp
        elif is_mla:
            # MLA: Q is (n_head * qk_head_dim) per rank, KV cache is the
            # latent (kv_lora_rank + qk_rope_head_dim) per token —
            # replicated across TP ranks (no // p), single buffer (no *2).
            attn_len = length if not pim else 1
            input_size = (
                (n_head // p) * attn_len * qk_head_dim * fp +
                mla_latent_dim * (kv_len if kv_len is not None else attn_len) * fp
            )
            weight_size = 0
            output_size = (n_head // p) * attn_len * v_head_dim * fp
        elif not pim:
            input_size = (
                (n_head // p) * length * head_dim * fp +
                (kv_head // p) * kv_len * head_dim * fp * 2
            )
            weight_size = 0
            output_size = (n_head // p) * length * head_dim * fp
        else:
            input_size = (
                (n_head // p) * 1 * head_dim * fp +
                (kv_head // p) * 1 * head_dim * fp * 2
            )
            weight_size = 0
            output_size = (n_head // p) * 1 * head_dim * fp

    # ----------------- QKV Projection (fused) -----------------
    elif layer_name == "qkv_proj":
        input_size = length * n_embd * fp
        weight_size = n_embd * ((q_dim + 2 * kv_dim) // p) * weight_fp
        output_size = length * ((q_dim + 2 * kv_dim) // p) * fp

    # ----------------- MLA A/B projections (DeepSeek V2/V3) -----------------
    elif layer_name == "fused_qkv_a_proj":
        # MergedColumnParallelLinear: 7168 → q_lora_rank + kv_lora_rank
        # + qk_rope_head_dim, output split across TP.
        input_size = length * n_embd * fp
        weight_size = n_embd * (a_proj_out // p) * weight_fp
        output_size = length * (a_proj_out // p) * fp

    elif layer_name == "q_a_layernorm":
        input_size = length * (q_lora_rank or 0) * fp
        weight_size = (q_lora_rank or 0) * fp
        output_size = input_size

    elif layer_name == "q_b_proj":
        # ColumnParallelLinear: q_lora_rank → n_head * qk_head_dim,
        # output split across TP.
        input_size = length * (q_lora_rank or 0) * fp
        weight_size = (q_lora_rank or 0) * (q_total // p) * weight_fp
        output_size = length * (q_total // p) * fp

    elif layer_name == "kv_a_layernorm":
        input_size = length * kv_lora_rank * fp
        weight_size = kv_lora_rank * fp
        output_size = input_size

    elif layer_name == "kv_b_proj":
        # ColumnParallelLinear: kv_lora_rank → n_head * (qk_nope + v).
        kv_b_out = n_head * (qk_nope_head_dim + v_head_dim)
        input_size = length * kv_lora_rank * fp
        weight_size = kv_lora_rank * (kv_b_out // p) * weight_fp
        output_size = length * (kv_b_out // p) * fp

    elif layer_name == "o_proj":
        if is_mla:
            input_size = length * (v_total // p) * fp
            weight_size = (v_total // p) * n_embd * weight_fp
            output_size = length * n_embd * fp
        else:
            input_size = length * (q_dim // p) * fp
            weight_size = (q_dim // p) * n_embd * weight_fp
            output_size = length * n_embd * fp

    # ----------------- V4 MLA projections (DeepSeek-V4) -----------------
    elif layer_name == "fused_wqa_wkv":
        # MergedColumnParallelLinear: hidden -> q_lora_rank + head_dim.
        # Smaller than V3's fused_qkv_a_proj because V4 drops the
        # separate kv_b path (KV stays in latent until the MLA kernel).
        input_size = length * n_embd * fp
        weight_size = n_embd * (v4_a_proj_out // p) * weight_fp
        output_size = length * (v4_a_proj_out // p) * fp

    elif layer_name == "q_norm":
        # RMSNorm on the q_lora_rank stream. Replicated, not TP-sharded.
        input_size = length * v4_q_lora_rank * fp
        weight_size = v4_q_lora_rank * fp
        output_size = input_size

    elif layer_name == "wq_b":
        # ColumnParallelLinear: q_lora_rank -> n_head * head_dim,
        # output split across TP.
        input_size = length * v4_q_lora_rank * fp
        q_total_v4 = n_head * v4_head_dim
        weight_size = v4_q_lora_rank * (q_total_v4 // p) * weight_fp
        output_size = length * (q_total_v4 // p) * fp

    elif layer_name == "kv_norm":
        # RMSNorm on the KV latent (head_dim wide in V4). Replicated.
        input_size = length * v4_head_dim * fp
        weight_size = v4_head_dim * fp
        output_size = input_size

    elif layer_name == "wo_a":
        # First half of grouped low-rank o_proj. Per-rank shape:
        # for each of o_groups groups, a (n_head*head_dim/o_groups) ->
        # (o_lora_rank // p) weight matrix. ColumnParallel splits the
        # o_lora_rank dim across TP; the bmm runs across o_groups.
        input_per_group = (n_head * v4_head_dim) // max(v4_o_groups, 1)
        weight_size = v4_o_groups * input_per_group * (v4_o_lora_rank // p) * weight_fp
        # Activation flow: input is the full n_head*head_dim attention
        # output (gathered across groups before the bmm), output is
        # o_groups * (o_lora_rank // p) per token.
        input_size = length * (n_head * v4_head_dim) * fp
        output_size = length * (v4_o_groups * v4_o_lora_rank // p) * fp

    elif layer_name == "wo_b":
        # Second half: RowParallelLinear, o_groups*o_lora_rank -> hidden.
        # Input dim is sharded across TP, output is full hidden.
        input_size = length * (v4_o_groups * v4_o_lora_rank // p) * fp
        weight_size = (v4_o_groups * v4_o_lora_rank // p) * n_embd * weight_fp
        output_size = length * n_embd * fp

    # ----------------- V4 Indexer subcomponents -----------------
    # Per-layer-conditional: only present where compress_ratios[i] == 4.
    # All projections are ReplicatedLinear / LayerNorm — not TP-sharded.
    elif layer_name == "idx_wq_b":
        # ReplicatedLinear: q_lora_rank -> index_n_heads * index_head_dim.
        idx_out = v4_index_n_heads * v4_index_head_dim
        input_size = length * v4_q_lora_rank * fp
        weight_size = v4_q_lora_rank * idx_out * weight_fp
        output_size = length * idx_out * fp

    elif layer_name == "idx_weights_proj":
        # ReplicatedLinear: hidden -> index_n_heads.
        input_size = length * n_embd * fp
        weight_size = n_embd * v4_index_n_heads * weight_fp
        output_size = length * v4_index_n_heads * fp

    elif layer_name == "idx_k_norm":
        # LayerNorm on index_head_dim. Note: vLLM uses nn.LayerNorm here,
        # not RMSNorm — so the bias term is present, but it shares the
        # same per-element size as the scale, doubling the weight.
        input_size = length * v4_index_head_dim * fp
        weight_size = 2 * v4_index_head_dim * fp        # scale + bias
        output_size = input_size

    elif layer_name in ("idx_compressor", "idx_indexer_op"):
        # Kernel-only: DeepseekCompressor and SparseAttnIndexer carry
        # no learnable weights — their state lives in the cache modules.
        # Activation shape passes through hidden_size.
        input_size = length * n_embd * fp
        weight_size = 0
        output_size = length * n_embd * fp

    elif layer_name == "gate_up_proj":
        input_size = length * n_embd * fp
        weight_size = n_embd * 2 * (ffn_dim // p) * weight_fp
        output_size = length * 2 * (ffn_dim // p) * fp

    elif layer_name == "act_fn":
        input_size = length * 2 * (ffn_dim // p) * fp
        weight_size = 0
        output_size = length * (ffn_dim // p) * fp

    elif layer_name == "down_proj":
        input_size = length * (ffn_dim // p) * fp
        weight_size = (ffn_dim // p) * n_embd * weight_fp
        output_size = length * n_embd * fp

    elif layer_name == "sampler":
        input_size = length * (vocab_size // p) * fp
        weight_size = 0
        output_size = length * 4  # int32 token IDs

    elif layer_name == "moe":
        experts_per_rank = num_local_experts // p
        input_size = length * n_embd * fp
        # Router gate stays at compute dtype (replicated, not quantized in
        # vLLM's fp8 scheme). Routed and shared experts use ``moe_weight_fp``
        # — distinct from ``weight_fp`` for models where the routed experts
        # are quantized more aggressively than the dense path (DeepSeek-V4:
        # fp8 dense + fp4 experts). For every existing model
        # (Mixtral / Qwen3-MoE / DeepSeek-R1 / phi-mini-MoE), the resolver
        # collapses ``moe_weight_fp == weight_fp`` so behaviour is unchanged.
        # Shared experts are not distributed by EP (every EP rank keeps the
        # shared expert), but they ARE TP-sharded along the intermediate dim
        # like the routed experts when moe_tp>1 (EP-off / partial-EP).
        gate_w = n_embd * num_local_experts * fp
        moe_ffn_shard = moe_ffn_dim // max(1, int(moe_tp))  # intra-expert TP shard (tp//ep; 1 when ep==tp)
        routed_w = experts_per_rank * 3 * n_embd * moe_ffn_shard * moe_weight_fp
        shared_w = n_shared_experts * 3 * n_embd * moe_ffn_shard * moe_weight_fp
        weight_size = gate_w + routed_w + shared_w
        output_size = length * n_embd * fp

    # ----------------- LM Head -----------------
    elif layer_name == "lm_head":
        input_size = length * n_embd * fp
        weight_size = n_embd * (vocab_size // p) * weight_fp
        output_size = length * (vocab_size // p) * fp

    else:
        raise ValueError(f"No matching layer name {layer_name} found for model {model}.")

    return input_size, weight_size, output_size
