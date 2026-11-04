"""Per-format expert weights for native MoE component acquisition.

A format is chosen by the routed experts' own vLLM quant method
(``runner.routed_experts.quant_method``), never by the model-level quant
config: vLLM's ``Fp8Config`` gives an expert on its ignore list an
``UnquantizedFusedMoEMethod`` (``RoutedExperts._get_quant_method``), and
those experts run unquantized.

Each format mirrors the three per-format steps vLLM 0.31 keeps in
``fused_moe/oracle/`` -- ``select_<fmt>_moe_backend``,
``convert_to_<fmt>_moe_kernel_format`` and ``make_<fmt>_moe_quant_config``
-- and supplies what acquisition needs around them: the seeded weights in
checkpoint layout, their fp32 values for the output check, the contract
fields and the DP1 prepare step. For a rebuilt shared expert it also says
which linear quant methods are admitted and how their checkpoint is seeded
and loaded. A routed method no format matches keeps the live path; adding a
format is one class in ``FORMATS``.
"""
from dataclasses import dataclass

# Relative Frobenius error allowed between one kernel call and the fp32
# reference over the same dequantized weights. What the two cannot share is
# the rounding of the kernel's own activation quantizations (input and
# intermediate), expected to be a few percent; a wrong layout, scale or expert
# map is O(1). The shared-expert check uses it for every format.
OUTPUT_TOLERANCE = 0.1


@dataclass
class ExpertWeights:
    """One rank's experts as the checkpoint stores them.

    ``w13`` stacks the gate and up projections, each with its own scale grid;
    ``scales`` is the format's own record of them (``Fp8Scales``), None when
    unquantized. ``clone`` copies all of it for another weight bank.
    """
    w13: object
    w2: object
    scales: object = None

    def clone(self):
        return ExpertWeights(self.w13.clone(), self.w2.clone(),
                             None if self.scales is None else self.scales.clone())

    def nbytes(self):
        return (sum(t.numel() * t.element_size() for t in (self.w13, self.w2))
                + (0 if self.scales is None else self.scales.nbytes()))


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


class UnquantizedExperts:
    """Experts and linears at the model's floating dtype."""
    name = 'unquantized'
    quantized = False
    # Not checked against fp32: the unquantized path has never carried one.
    output_tolerance = None

    def __init__(self, vllm_config, quant_method=None):
        self.method = quant_method

    @staticmethod
    def matches(quant_method):
        from vllm.model_executor.layers.fused_moe import UnquantizedFusedMoEMethod
        return isinstance(quant_method, UnquantizedFusedMoEMethod)

    @staticmethod
    def matches_linear(quant_method):
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        return isinstance(quant_method, UnquantizedLinearMethod)

    def admit(self, target):
        pass

    def checkpoint(self, w13, w2, shards):
        return ExpertWeights(w13, w2)

    def check_weights(self, config, weights):
        import torch
        w13, w2 = weights.w13, weights.w2
        if config.in_dtype != w13.dtype or w13.dtype != w2.dtype or w13.device != w2.device:
            raise ValueError("Unquantized weights and activation dtype/device must agree")
        if w13.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise NotImplementedError("Quantized experts require a quantization-aware adapter")

    # vLLM's three oracle steps (fused_moe/oracle/unquantized.py).
    def select_backend(self, config, weights):
        from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
            select_unquantized_moe_backend)
        backend, cls = select_unquantized_moe_backend(config)
        return backend, cls, None

    def convert(self, backend, config, weights, selection):
        from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
            convert_to_unquantized_kernel_format)
        w13, w2 = convert_to_unquantized_kernel_format(backend, config, weights.w13, weights.w2)
        return w13, w2, None

    def quant_config(self, backend, kernel_scales, selection):
        from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
        return FUSED_MOE_UNQUANTIZED_CONFIG

    def prepare(self):
        """vLLM's DP1 prepare when it runs inside the timed call, else None."""
        return None

    def contract(self, weights, backend, selection, experts):
        return {}

    def recipe(self):
        return None

    def dense_expert(self, weights, expert, shards):
        return weights.w13[expert].float(), weights.w2[expert].float()

    def prepare_dp1(self, finalize, hidden, topk_weights, ids, config, mapping, experts):
        """vLLM's DP1 prepare as the experts call it, and whether it quantizes."""
        from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
        prepared = finalize.prepare(hidden, topk_weights, ids, config.num_experts, mapping,
                                    False, FUSED_MOE_UNQUANTIZED_CONFIG)
        return prepared, False

    def quantized_input(self, prepared, hidden):
        raise NotImplementedError('Unquantized experts take no quantized input')

    # A rebuilt shared expert's linears.
    def linear_checkpoint(self, linear, weight):
        return weight, None

    def linear_load(self, linear, weight, scale, shard_ids):
        linear.weight.weight_loader(linear.weight, weight, *shard_ids)

    def linear_dense(self, weight, scale, linear):
        return weight.float()


class Fp8Experts(UnquantizedExperts):
    """vLLM's ``Fp8MoEMethod`` experts and ``Fp8LinearMethod`` linears, block
    scales and dynamic activations only.

    The oracle picks the backend for the *target* configuration -- at EP > 1
    on Hopper that is FlashInfer CUTLASS, not the Triton kernel an EP1 engine
    would run -- converts the checkpoint layout into the kernel's and builds
    the quantization config. The DP1 prepare step either quantizes the input
    or, for experts that quantize it themselves, passes it through; the
    deployment runs it, so it is timed with the kernel. The FP8 dtype is the
    platform's (``current_platform.fp8_dtype()``), as vLLM's own
    ``per_block_cast_to_fp8`` writes it.
    """
    name = 'fp8'
    quantized = True
    output_tolerance = OUTPUT_TOLERANCE

    def __init__(self, vllm_config, quant_method=None):
        super().__init__(vllm_config, quant_method)
        hf_quant = getattr(vllm_config.model_config.hf_config, 'quantization_config', None)
        # Power-of-two scales, as the checkpoint stores them under ue8m0.
        self.ue8m0 = isinstance(hf_quant, dict) and hf_quant.get('scale_fmt') == 'ue8m0'

    @staticmethod
    def matches(quant_method):
        from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod
        return isinstance(quant_method, Fp8MoEMethod)

    @staticmethod
    def matches_linear(quant_method):
        from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
        return (isinstance(quant_method, Fp8LinearMethod) and quant_method.block_quant
                and not quant_method.act_q_static)

    @staticmethod
    def fp8_dtype():
        from vllm.platforms import current_platform
        return current_platform.fp8_dtype()

    def admit(self, target):
        quant = self.method.quant_config
        if quant.weight_block_size is None or quant.activation_scheme != 'dynamic':
            raise NotImplementedError("FP8 experts need block scales and dynamic activations; "
                                      "per-tensor and static schemes are not admitted")
        if target.dp > 1:
            raise NotImplementedError("FP8 experts at DP > 1 quantize before the dispatch "
                                      "all-gather, which this expert region does not time")

    def checkpoint(self, w13, w2, shards):
        """Seeded weights in the FP8 checkpoint layout, one rank's experts.

        Every weight block gets the dequantization scale ``amax / 448`` that a
        serialized checkpoint carries -- written by vLLM's own
        ``per_block_cast_to_fp8``, rounded to a power of two when the
        checkpoint declares ``scale_fmt: ue8m0``. ``w13`` stacks ``shards``
        projections (gate and up) and each has its own block grid, as in the
        scale tensor ``Fp8MoEMethod.create_weights`` allocates; that matters
        whenever the intermediate size is not a multiple of the block. Experts
        are cast one at a time to bound the temporary memory.
        """
        import torch
        from vllm.utils.deep_gemm import per_block_cast_to_fp8

        quant = self.method.quant_config
        block = list(quant.weight_block_size)

        def cast(matrix, parts):
            cast_parts = [per_block_cast_to_fp8(part, block, use_ue8m0=self.ue8m0)
                          for part in matrix.chunk(parts, dim=0)]
            return torch.cat([v for v, _ in cast_parts]), torch.cat([s for _, s in cast_parts])

        def cast_all(weights, parts):
            values, scales = zip(*(cast(w, parts) for w in weights))
            return torch.stack(values), torch.stack(scales)

        q13, s13 = cast_all(w13, shards)
        q2, s2 = cast_all(w2, 1)
        return ExpertWeights(q13, q2, Fp8Scales(s13, s2, tuple(block),
                                                bool(quant.is_checkpoint_fp8_serialized)))

    def check_weights(self, config, weights):
        import torch
        w13, w2, fp8 = weights.w13, weights.w2, self.fp8_dtype()
        if (w13.dtype != fp8 or w2.dtype != w13.dtype or w13.device != w2.device
                or config.in_dtype not in (torch.float16, torch.bfloat16)):
            raise ValueError(f"FP8 experts need {fp8} weights on one device and 16-bit activations")

    # vLLM's three oracle steps (fused_moe/oracle/fp8.py), keyed exactly as
    # Fp8MoEMethod keys them.
    def select_backend(self, config, weights):
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
            resolve_fp8_moe_weight_block_shape, select_fp8_moe_backend)
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            GroupShape, create_fp8_quant_key, kFp8Dynamic128Sym)

        scales = weights.scales
        block, refine = resolve_fp8_moe_weight_block_shape(
            config, list(scales.weight_block_size), kFp8Dynamic128Sym,
            scales.checkpoint_serialized)
        if refine is not None:
            raise NotImplementedError("Refined FP8 block scales need the upsampling vLLM's loader applies")
        weight_key = create_fp8_quant_key(static=True, group_shape=GroupShape(*block))
        backend, cls = select_fp8_moe_backend(config=config, weight_key=weight_key,
                                              activation_key=kFp8Dynamic128Sym,
                                              allow_vllm_cutlass=False)
        return backend, cls, dict(block=list(block), weight_key=weight_key,
                                  activation_key=kFp8Dynamic128Sym)

    def convert(self, backend, config, weights, selection):
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import convert_to_fp8_moe_kernel_format

        layer = self._layer_view(config, weights.scales.weight_block_size)
        try:
            w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
                backend, layer, weights.w13, weights.w2, weights.scales.w13, weights.scales.w2,
                None, None)
        except AttributeError as exc:
            raise NotImplementedError(
                f"FP8 backend {backend.value} converts weights through layer state "
                f"this region does not provide: {exc}") from exc
        return w13, w2, (w13_scale, w2_scale, layer)

    @staticmethod
    def _layer_view(config, weight_block_size):
        """The ``RoutedExperts`` state vLLM's FP8 weight conversion reads.

        Conversion for a backend that needs more raises ``AttributeError``,
        which ``convert`` reports as an unadmitted backend instead of guessing
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

    def quant_config(self, backend, kernel_scales, selection):
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import make_fp8_moe_quant_config
        w13_scale, w2_scale, layer = kernel_scales
        return make_fp8_moe_quant_config(
            fp8_backend=backend, w1_scale=w13_scale, w2_scale=w2_scale,
            a1_scale=None, a2_scale=None, block_shape=selection['block'], layer=layer)

    def prepare(self):
        # vLLM's DP1 prepare. The caller verifies outside timing that it only
        # quantizes or passes through the input (moe_components).
        from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
            MoEPrepareAndFinalizeNoDPEPModular)
        return MoEPrepareAndFinalizeNoDPEPModular()

    def contract(self, weights, backend, selection, experts):
        # The DP1 prepare runs inside the timed call, so only finalize is left
        # out; the quantization keys are the ones vLLM selected with.
        scales = weights.scales
        return dict(
            quantization=dict(method=self.name, weight_block_size=list(scales.weight_block_size),
                              kernel_block_shape=selection['block'],
                              weight_key=str(selection['weight_key']),
                              activation_key=str(selection['activation_key']),
                              checkpoint_serialized=scales.checkpoint_serialized,
                              scale_layout="per_block_weight_scale_inv"),
            input_quantization=("experts" if experts.expects_unquantized_inputs else "prepare"),
            prepare_finalize="vllm.model_executor.layers.fused_moe.prepare_finalize."
                             "no_dp_ep.MoEPrepareAndFinalizeNoDPEPModular",
            excluded=["gate_routing", "collectives", "finalize", "shared_experts", "cpu_time"])

    def recipe(self):
        return dict(values='seeded_randn_0.01', quantization='per_block_cast_to_fp8',
                    scale_format='ue8m0' if self.ue8m0 else 'float32',
                    output_check=dict(reference='fp32_dequantized_local_experts',
                                      relative_frobenius_tolerance=self.output_tolerance))

    @staticmethod
    def _dequantize(values, scales, block, parts=1):
        """fp32 weights from FP8 values and per-block ``weight_scale_inv``.

        ``parts`` stacked projections each carry their own block grid (see
        ``checkpoint``).
        """
        import torch
        if parts > 1:
            return torch.cat([Fp8Experts._dequantize(v, s, block) for v, s in
                              zip(values.chunk(parts, dim=0), scales.chunk(parts, dim=0))])
        rows, cols = values.shape[-2:]
        scale = scales.repeat_interleave(block[0], dim=-2)[..., :rows, :]
        return values.float() * scale.repeat_interleave(block[1], dim=-1)[..., :cols]

    def dense_expert(self, weights, expert, shards):
        scales, block = weights.scales, weights.scales.weight_block_size
        return (self._dequantize(weights.w13[expert], scales.w13[expert], block, shards),
                self._dequantize(weights.w2[expert], scales.w2[expert], block))

    def prepare_dp1(self, finalize, hidden, topk_weights, ids, config, mapping, experts):
        quantizes = not experts.expects_unquantized_inputs
        prepared = finalize.prepare(hidden, topk_weights, ids, config.num_experts, mapping,
                                    False, experts.quant_config, defer_input_quant=not quantizes)
        return prepared, quantizes

    def quantized_input(self, prepared, hidden):
        """Whether a DP1 prepare's output is a plain FP8 quantization of ``hidden``."""
        return (prepared[0].dtype == self.fp8_dtype() and prepared[0].shape == hidden.shape
                and prepared[1] is not None)

    def linear_checkpoint(self, linear, weight):
        """``(values, scale)`` as a checkpoint stores this linear's weight.

        Block-FP8 values carry one ``amax / 448`` dequantization scale per
        block, written by vLLM's ``per_block_cast_to_fp8`` (power-of-two for
        ``scale_fmt: ue8m0``), as for the routed experts.
        """
        from vllm.utils.deep_gemm import per_block_cast_to_fp8
        return per_block_cast_to_fp8(weight, list(linear.weight_block_size), use_ue8m0=self.ue8m0)

    def linear_load(self, linear, weight, scale, shard_ids):
        # vLLM's own loaders place each shard and its scale block rows.
        linear.weight.weight_loader(linear.weight, weight, *shard_ids)
        linear.weight_scale_inv.weight_loader(linear.weight_scale_inv, scale, *shard_ids)

    def linear_dense(self, weight, scale, linear):
        return self._dequantize(weight, scale, linear.weight_block_size)


FORMATS = (UnquantizedExperts, Fp8Experts)


def expert_format(quant_method, vllm_config):
    """The format of routed experts with this vLLM quant method, or None."""
    for fmt in FORMATS:
        if fmt.matches(quant_method):
            return fmt(vllm_config, quant_method)
    return None


def linear_format(quant_method, vllm_config):
    """The format of a linear with this vLLM quant method, or None."""
    for fmt in FORMATS:
        if fmt.matches_linear(quant_method):
            return fmt(vllm_config)
    return None


def expert_reference(fmt, hidden, ids, topk_weights, weights, place, cfg):
    """fp32 output of this rank's own experts for a routed batch.

    Remote experts contribute nothing, as on the rank; each local expert is
    turned dense (``fmt.dense_expert``) only when a token selects it.
    """
    import torch
    import torch.nn.functional as F
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    gated = {MoEActivation.SILU: F.silu, MoEActivation.GELU: F.gelu,
             MoEActivation.GELU_TANH: lambda x: F.gelu(x, approximate='tanh')}
    if cfg.activation not in gated or cfg.w13_num_shards != 2:
        raise NotImplementedError(f'The expert output check covers gated SiLU/GELU experts, not {cfg.activation}')
    act, inter = gated[cfg.activation], cfg.intermediate_size_per_partition
    local = torch.tensor(place.expert_map, device=hidden.device)[ids.long()]
    reference = torch.zeros(hidden.shape, device=hidden.device, dtype=torch.float32)
    for expert in range(weights.w13.shape[0]):
        rows, slots = (local == expert).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        w13, w2 = fmt.dense_expert(weights, expert, cfg.w13_num_shards)
        gate_up = hidden[rows].float() @ w13.T
        output = (act(gate_up[:, :inter]) * gate_up[:, inter:]) @ w2.T
        reference.index_add_(0, rows, output * topk_weights[rows, slots].unsqueeze(1))
    return reference
