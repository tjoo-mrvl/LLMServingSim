"""Native modular expert region without DP2 or hardware-specific branches.

The caller provides a resolved vLLM config and checkpoint-layout local weights.
Only the post-dispatch expert region is measured; gate, transport and finalize
remain separate. Backend selection and workspace aliasing belong to vLLM.

FP8 weights arrive with their ``Fp8Scales`` and follow vLLM's own FP8 MoE
recipe (``Fp8MoEMethod``): the oracle picks the backend for the *target*
configuration -- at EP > 1 on Hopper that is FlashInfer CUTLASS, not the
Triton kernel an EP1 engine would run -- converts the checkpoint layout into
the kernel's and builds the quantization config. Their DP1 prepare step either
quantizes the input or, for experts that quantize it themselves, passes it
through; the deployment runs it, so it is timed with the kernel.
"""
from dataclasses import asdict, dataclass


@dataclass
class Fp8Scales:
    """The block scales that make a rank's FP8 expert weights a checkpoint.

    ``w13`` / ``w2`` are vLLM's ``weight_scale_inv`` tensors: one float32
    dequantization multiplier per ``weight_block_size`` block of the matching
    weight, in the layout ``Fp8MoEMethod.create_weights`` allocates.
    ``clone`` copies them for another weight bank, as the weights are copied.
    """
    w13: object
    w2: object
    weight_block_size: tuple
    checkpoint_serialized: bool = True

    def clone(self):
        return Fp8Scales(self.w13.clone(), self.w2.clone(), self.weight_block_size,
                         self.checkpoint_serialized)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.w13, self.w2))


def _fp8_layer_view(config, weight_block_size):
    """The ``RoutedExperts`` state vLLM's FP8 weight conversion reads.

    Conversion for a backend that needs more raises ``AttributeError``, which
    ``NativeExpertRegion`` reports as an unadmitted backend instead of guessing
    the missing state.
    """
    import torch

    layer = torch.nn.Module()
    layer.moe_config = config
    layer.weight_block_size = list(weight_block_size)
    layer.activation = config.activation
    layer.local_num_experts = config.num_local_experts
    layer.global_num_experts = config.num_experts
    layer.hidden_size = config.hidden_dim
    layer.intermediate_size_per_partition = config.intermediate_size_per_partition
    layer.orig_dtype = config.in_dtype
    return layer


def _select_fp8_backend(config, scales):
    """vLLM's FP8 backend for this target, keyed exactly as ``Fp8MoEMethod``."""
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        resolve_fp8_moe_weight_block_shape, select_fp8_moe_backend)
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape, create_fp8_quant_key, kFp8Dynamic128Sym)

    block, refine = resolve_fp8_moe_weight_block_shape(
        config, list(scales.weight_block_size), kFp8Dynamic128Sym,
        scales.checkpoint_serialized)
    if refine is not None:
        raise NotImplementedError("Refined FP8 block scales need the upsampling vLLM's loader applies")
    weight_key = create_fp8_quant_key(static=True, group_shape=GroupShape(*block))
    backend, cls = select_fp8_moe_backend(config=config, weight_key=weight_key,
                                          activation_key=kFp8Dynamic128Sym,
                                          allow_vllm_cutlass=False)
    return backend, cls, list(block), weight_key, kFp8Dynamic128Sym


class NativeExpertRegion:
    def __init__(self, config, w13, w2, fp8=None):
        import torch
        import vllm
        from vllm.model_executor.layers.fused_moe import modular_kernel as mk
        from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
        from vllm.model_executor.layers.fused_moe.expert_map_manager import determine_expert_map
        from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
            convert_to_unquantized_kernel_format, select_unquantized_moe_backend)

        parallel = config.moe_parallel_config
        if (config.has_bias or config.is_lora_enabled or parallel.enable_eplb
                or config.num_experts != config.num_logical_experts):
            raise NotImplementedError("Bias, LoRA, EPLB and redundant experts need separate adapters")
        if fp8 is None:
            if config.in_dtype != w13.dtype or w13.dtype != w2.dtype or w13.device != w2.device:
                raise ValueError("Unquantized weights and activation dtype/device must agree")
            if w13.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise NotImplementedError("Quantized experts require a quantization-aware adapter")
        elif (w13.dtype != torch.float8_e4m3fn or w2.dtype != w13.dtype or w13.device != w2.device
                or config.in_dtype not in (torch.float16, torch.bfloat16)):
            raise ValueError("FP8 experts need float8_e4m3fn weights on one device and 16-bit activations")
        if parallel.tp_size != 1 or parallel.pcp_size != 1:
            raise NotImplementedError("Tensor-sharded experts and PCP are not covered")
        local_e, mapping, _ = determine_expert_map(parallel.ep_size, parallel.ep_rank,
                                                  config.num_experts, "linear")
        inter = config.intermediate_size_per_partition
        if (config.num_local_experts != local_e
                or tuple(w13.shape) != (local_e, config.w13_num_shards * inter, config.hidden_dim)
                or tuple(w2.shape) != (local_e, config.hidden_dim, inter)):
            raise ValueError("Weights do not match the resolved local expert shape")
        if fp8 is None:
            backend, cls = select_unquantized_moe_backend(config)
        else:
            backend, cls, block, weight_key, activation_key = _select_fp8_backend(config, fp8)
        if cls is None or not issubclass(cls, mk.FusedMoEExpertsModular):
            raise NotImplementedError(f"Native backend {backend} needs a monolithic adapter")
        if cls.activation_format() != mk.FusedMoEActivationFormat.Standard:
            raise NotImplementedError(f"Native backend {backend} needs dispatched token metadata")
        self.config = config
        self.mapping = None if mapping is None else mapping.to(w13.device)
        if fp8 is None:
            self.w13, self.w2 = convert_to_unquantized_kernel_format(backend, config, w13, w2)
            self.experts = cls(config, FUSED_MOE_UNQUANTIZED_CONFIG)
            self.prepare = None
        else:
            from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
                convert_to_fp8_moe_kernel_format, make_fp8_moe_quant_config)
            from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
                MoEPrepareAndFinalizeNoDPEPModular)
            layer = _fp8_layer_view(config, fp8.weight_block_size)
            try:
                self.w13, self.w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
                    backend, layer, w13, w2, fp8.w13, fp8.w2, None, None)
            except AttributeError as exc:
                raise NotImplementedError(
                    f"FP8 backend {backend.value} converts weights through layer state "
                    f"this region does not provide: {exc}") from exc
            quant_config = make_fp8_moe_quant_config(
                fp8_backend=backend, w1_scale=w13_scale, w2_scale=w2_scale,
                a1_scale=None, a2_scale=None, block_shape=block, layer=layer)
            self.experts = cls(moe_config=config, quant_config=quant_config)
            # vLLM's DP1 prepare. The caller verifies outside timing that it
            # only quantizes or passes through the input (moe_components).
            self.prepare = MoEPrepareAndFinalizeNoDPEPModular()
        self.impl = mk.FusedMoEKernelModularImpl(None, self.experts)
        self.contract = dict(
            schema="native-modular-expert-region", vllm_version=str(vllm.__version__),
            backend=backend.value, expert_class=f"{cls.__module__}.{cls.__name__}",
            expert_parallel_config=asdict(parallel), global_experts=config.num_experts,
            global_top_k=config.experts_per_token, local_experts=local_e,
            expert_map=None if mapping is None else mapping.tolist(),
            hidden_dim=config.hidden_dim, intermediate_size=inter,
            activation=str(config.activation), input_dtype=str(config.in_dtype),
            weight_dtype=str(w13.dtype), workspace="native_shared_manager",
            excluded=["gate_routing", "collectives", "prepare_finalize", "shared_experts", "cpu_time"])
        if fp8 is not None:
            # The DP1 prepare runs inside the timed call, so only finalize is
            # left out; the quantization keys are the ones vLLM selected with.
            self.contract.update(
                quantization=dict(method="fp8", weight_block_size=list(fp8.weight_block_size),
                                  kernel_block_shape=block, weight_key=str(weight_key),
                                  activation_key=str(activation_key),
                                  checkpoint_serialized=fp8.checkpoint_serialized,
                                  scale_layout="per_block_weight_scale_inv"),
                input_quantization=("experts" if self.experts.expects_unquantized_inputs
                                    else "prepare"),
                prepare_finalize="vllm.model_executor.layers.fused_moe.prepare_finalize."
                                 "no_dp_ep.MoEPrepareAndFinalizeNoDPEPModular",
                excluded=["gate_routing", "collectives", "finalize", "shared_experts", "cpu_time"])

    def bind(self, hidden, ids, weights, *, output_alias=None):
        """Validate outside measurement; preserve native allocation inside it."""
        import torch
        from vllm.v1.worker.workspace import current_workspace_manager

        cfg = self.config
        if hidden.ndim != 2 or not 1 <= hidden.shape[0] <= cfg.max_num_tokens:
            raise ValueError("Gathered token count is outside the configured measurement range")
        if hidden.shape[1] != cfg.hidden_dim or hidden.dtype != cfg.in_dtype:
            raise ValueError("Hidden shape/dtype differs from the resolved contract")
        if ids.shape != (hidden.shape[0], cfg.experts_per_token) or weights.shape != ids.shape:
            raise ValueError("Routing must retain the complete global top-k width")
        if ids.dtype not in (torch.int32, torch.int64) or weights.dtype != torch.float32:
            raise ValueError("Unexpected routing dtype")
        if any(v.device != self.w13.device for v in (hidden, ids, weights)):
            raise ValueError("All inputs must belong to the measurement device")
        if ids.min().item() < 0 or ids.max().item() >= cfg.num_experts:
            raise ValueError("Global expert ID out of range")
        ordered = ids.sort(dim=1).values
        if cfg.experts_per_token > 1 and (ordered[:, 1:] == ordered[:, :-1]).any().item():
            raise ValueError("A token cannot select the same expert twice")
        if not torch.isfinite(hidden).all().item() or not torch.isfinite(weights).all().item():
            raise ValueError("Non-finite measurement stimulus")
        if current_workspace_manager() is None:
            raise RuntimeError("Initialize the native workspace manager before binding")

        def experts(a1q, a1q_scale):
            return self.impl._fused_experts(
                in_dtype=hidden.dtype, a1q=a1q, a1q_scale=a1q_scale,
                w1=self.w13, w2=self.w2, topk_weights=weights, topk_ids=ids,
                activation=cfg.activation, global_num_experts=cfg.num_experts,
                local_num_experts=cfg.num_local_experts, expert_map=self.mapping,
                apply_router_weight_on_input=False, expert_tokens_meta=None,
                output_alias=output_alias)

        if self.prepare is None:
            def compute():
                return experts(hidden, None)
            return compute

        prepare, quant_config = self.prepare, self.experts.quant_config
        defer = self.experts.expects_unquantized_inputs

        def compute():
            # Quantizes the input to FP8, or hands it through when the experts
            # quantize it themselves. Token order and metadata are unchanged.
            a1q, a1q_scale, _, _, _ = prepare.prepare(
                hidden, weights, ids, cfg.num_experts, self.mapping, False,
                quant_config, defer_input_quant=defer)
            return experts(a1q, a1q_scale)

        return compute
