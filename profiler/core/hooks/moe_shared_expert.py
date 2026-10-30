"""One EP rank's shared expert, rebuilt at its tensor-parallel width.

vLLM runs a model's shared expert -- a gated MLP the MoE runner holds as
``runner._shared_experts._layer`` -- beside the routed experts. Then
``MoERunner.forward`` scales the routed output
(``_maybe_apply_routed_scale_to_output``), adds the two and reduces the sum
over TP once. Unlike the routed experts, the shared MLP is tensor-parallel:
every TP rank of a full-EP target holds 1/TP of its intermediate width.

The acquisition engine runs at TP1, so its module has the full width.
``SharedExpertCopy`` copies the module and rebuilds its two linears at
width/TP with the same vLLM classes, quantization config and prefix. vLLM
then picks the quantization method and GEMM kernel that a TP rank's shapes
get. The module's own ``forward`` and ``act_fn`` run unchanged.

Admission goes by structure, never by model family. Required: a
``gate_up_proj`` ``MergedColumnParallelLinear`` with two equal halves, an
``act_fn`` without weights, and a ``down_proj`` ``RowParallelLinear`` over one
half that leaves the reduction to the runner. Nothing else may hold weights,
and there is no bias. Both linears must be TP-sharded and unquantized, or
block-FP8 with dynamic activations. Anything else raises with the mismatch
named. Sequence-parallel MoE replicates the shared expert and is refused.
"""
import copy
import logging
import math

import torch

from .moe_components import FP8_OUTPUT_TOLERANCE, _dequantize


def _qualname(cls):
    return f'{cls.__module__}.{cls.__qualname__}'


def _quantization(linear, problems):
    """The linear's FP8 method, None when unquantized, or a recorded problem."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

    method = linear.quant_method
    if isinstance(method, UnquantizedLinearMethod):
        return None
    if isinstance(method, Fp8LinearMethod) and method.block_quant and not method.act_q_static:
        return method
    problems.append(f'{linear.prefix} uses {type(method).__name__}; only unquantized and '
                    'block-FP8 linears with dynamic activations are admitted')
    return None


def admit(runner, target, hidden_dim):
    """The runner's shared MLP, or ``NotImplementedError`` naming every mismatch."""
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear

    layer = getattr(runner._shared_experts, '_layer', None)
    if not isinstance(layer, torch.nn.Module):
        raise NotImplementedError('The MoE runner holds no shared-expert module')
    problems = []
    if target.sequence_parallel:
        problems.append('sequence-parallel MoE replicates it on every TP rank')
    if getattr(runner, 'shared_expert_gate', None) is not None:
        problems.append('a shared-expert gate scales its output')
    if set(layer._modules) != {'gate_up_proj', 'act_fn', 'down_proj'}:
        problems.append(f'its submodules {sorted(layer._modules)} are not exactly '
                        'gate_up_proj, act_fn and down_proj')
    elif (any(True for _ in layer.parameters(recurse=False))
            or any(True for _ in layer.buffers(recurse=False))
            or any(True for _ in layer.act_fn.parameters())
            or any(True for _ in layer.act_fn.buffers())):
        problems.append('it holds weights outside gate_up_proj and down_proj')
    else:
        gate_up, down = layer.gate_up_proj, layer.down_proj
        if (not isinstance(gate_up, MergedColumnParallelLinear)
                or len(gate_up.output_sizes) != 2
                or gate_up.output_sizes[0] != gate_up.output_sizes[1]
                or gate_up.gather_output or gate_up.input_size != hidden_dim):
            problems.append('gate_up_proj is not a MergedColumnParallelLinear of two equal '
                            'halves over the hidden size')
        elif (not isinstance(down, RowParallelLinear) or down.reduce_results
                or not down.input_is_parallel or down.input_size != gate_up.output_sizes[0]
                or down.output_size != hidden_dim):
            problems.append('down_proj is not a RowParallelLinear over one half that leaves '
                            'the reduction to the runner')
        else:
            for linear in (gate_up, down):
                if linear.has_bias:
                    problems.append(f'{linear.prefix} has a bias')
                if linear.disable_tp:
                    problems.append(f'{linear.prefix} is replicated across TP (disable_tp)')
                _quantization(linear, problems)
            if gate_up.output_sizes[0] % target.tp:
                problems.append(f'its intermediate size {gate_up.output_sizes[0]} does not '
                                f'divide by TP={target.tp}')
    if problems:
        raise NotImplementedError('Shared expert is not admitted: ' + '; '.join(problems))
    return layer


class SharedExpertCopy:
    """A TP rank's shared expert, plus the runner's native combine step.

    The timed call is what ``MoERunner.forward`` adds when a model has a
    shared expert: the MLP on the rank's local rows, the routed scale, and
    the add. The routed output itself is the ``experts`` component's work,
    so here it is a zero tensor of the right shape.
    """

    def __init__(self, vllm_config, runner, target, device, ue8m0):
        from vllm.config import set_current_vllm_config

        self.runner = runner
        layer = admit(runner, target, runner.moe_config.hidden_dim)
        gate_up, down = layer.gate_up_proj, layer.down_proj
        half = gate_up.output_sizes[0]
        width = half // target.tp
        # The same classes and constructor arguments at width/TP. The engine is
        # TP1, so the layers do not shard again. The current config matters:
        # Fp8LinearMethod reads the model dtype from it and selects its kernel
        # during create_weights.
        with set_current_vllm_config(vllm_config), torch.device(device):
            new_gate_up = type(gate_up)(
                gate_up.input_size, [width, width], bias=False,
                gather_output=gate_up.gather_output, skip_bias_add=gate_up.skip_bias_add,
                params_dtype=gate_up.params_dtype, quant_config=gate_up.quant_config,
                prefix=gate_up.prefix, return_bias=gate_up.return_bias,
                disable_tp=gate_up.disable_tp)
            new_down = type(down)(
                width, down.output_size, bias=False, input_is_parallel=down.input_is_parallel,
                skip_bias_add=down.skip_bias_add, params_dtype=down.params_dtype,
                reduce_results=down.reduce_results, quant_config=down.quant_config,
                prefix=down.prefix, return_bias=down.return_bias, disable_tp=down.disable_tp)
        for old, new in ((gate_up, new_gate_up), (down, new_down)):
            if type(new.quant_method) is not type(old.quant_method):
                raise ValueError(f'{old.prefix} rebuilt with {type(new.quant_method).__name__}, '
                                 f'not the engine\'s {type(old.quant_method).__name__}')
        # Same forward and act_fn; only the two linears differ. copy.copy shares
        # the module dicts, so give the copy its own before replacing entries.
        self.module = copy.copy(layer)
        self.module._modules = type(layer._modules)(layer._modules)
        self.module._parameters = type(layer._parameters)(layer._parameters)
        self.module._buffers = type(layer._buffers)(layer._buffers)
        self.module.gate_up_proj, self.module.down_proj = new_gate_up, new_down

        gen = torch.Generator(device=device).manual_seed(0)
        h, dtype = gate_up.input_size, gate_up.params_dtype

        def values(rows, cols):
            return torch.randn((rows, cols), device=device, dtype=dtype, generator=gen)*.01

        # Checkpoint layout: gate and up are separate tensors, each with its own
        # scale grid, loaded as shards 0 and 1 of gate_up_proj.
        self.checkpoint = dict(
            gate_up_proj=[self._checkpoint(new_gate_up, values(width, h), ue8m0),
                          self._checkpoint(new_gate_up, values(width, h), ue8m0)],
            down_proj=[self._checkpoint(new_down, values(h, width), ue8m0)])
        for name, linear in (('gate_up_proj', new_gate_up), ('down_proj', new_down)):
            shards = self.checkpoint[name]
            for shard_id, (weight, scale) in enumerate(shards):
                # vLLM's own loaders place each shard and its scale block rows.
                ids = (shard_id,) if len(shards) > 1 else ()
                linear.weight.weight_loader(linear.weight, weight, *ids)
                if scale is not None:
                    linear.weight_scale_inv.weight_loader(linear.weight_scale_inv, scale, *ids)
            # What vLLM's model loader does once every weight is in place.
            linear.quant_method.process_weights_after_loading(linear)
            if hasattr(linear, 'update_param_tp_status'):
                linear.update_param_tp_status()

        def record(name, linear):
            method = linear.quant_method
            weight, scale = self.checkpoint[name][0]
            return dict(linear_class=_qualname(type(linear)),
                        quant_method=_qualname(type(method)),
                        kernel=(_qualname(type(method.fp8_linear))
                                if hasattr(method, 'fp8_linear') else None),
                        weight_dtype=str(weight.dtype),
                        weight_block_size=(None if scale is None
                                           else list(linear.weight_block_size)))

        self.contract = dict(
            schema='native-shared-expert-mlp', module_class=_qualname(type(layer)),
            act_class=_qualname(type(layer.act_fn)),
            linears=dict(gate_up_proj=record('gate_up_proj', new_gate_up),
                         down_proj=record('down_proj', new_down)),
            hidden_dim=h, intermediate_size=width, unsharded_intermediate_size=half,
            tensor_parallel=target.tp, input_dtype=str(dtype),
            weight_bytes=sum(t.numel()*t.element_size() for shards in self.checkpoint.values()
                             for pair in shards for t in pair if t is not None),
            routed_scaling_factor=float(runner.routed_scaling_factor),
            boundary='shared_mlp_routed_scale_and_add',
            weight_recipe=dict(values='seeded_randn_0.01', loading='native_weight_loader',
                               output_check=dict(reference='fp32_dequantized_weights',
                                                 relative_frobenius_tolerance=FP8_OUTPUT_TOLERANCE)))

    @staticmethod
    def _checkpoint(linear, weight, ue8m0):
        """``(values, scale)`` as a checkpoint stores this linear's weight.

        Block-FP8 values carry one ``amax / 448`` dequantization scale per
        block, written by vLLM's ``per_block_cast_to_fp8`` (power-of-two for
        ``scale_fmt: ue8m0``), as for the routed experts.
        """
        if _quantization(linear, []) is None:
            return weight, None
        from vllm.utils.deep_gemm import per_block_cast_to_fp8
        return per_block_cast_to_fp8(weight, list(linear.weight_block_size), use_ue8m0=ue8m0)

    @torch.inference_mode()
    def check_output(self, hidden):
        """One call against fp32 arithmetic over the same dequantized weights.

        Runs once, before any timing. FP8 linears quantize their inputs, which
        the reference does not, so a few percent is expected; a wrong layout,
        scale or shard placement is of order one.
        """
        def dense(weight, scale, block):
            return weight.float() if scale is None else _dequantize(weight, scale, block)

        def matrix(name, linear):
            block = getattr(linear, 'weight_block_size', None)
            return torch.cat([dense(w, s, block) for w, s in self.checkpoint[name]])

        act = self.module.act_fn
        native = getattr(act, 'forward_native', act)
        gate_up = hidden.float() @ matrix('gate_up_proj', self.module.gate_up_proj).T
        reference = native(gate_up) @ matrix('down_proj', self.module.down_proj).T
        error = ((self.module(hidden).float() - reference).norm() / reference.norm()).item()
        if not math.isfinite(error) or error > FP8_OUTPUT_TOLERANCE:
            raise ValueError(f'Shared-expert output differs from the fp32 reference by {error:.4f} '
                             f'(relative Frobenius norm, tolerance {FP8_OUTPUT_TOLERANCE})')
        logging.getLogger(__name__).info(
            'Shared-expert output check: relative error %.4f (tolerance %s)',
            error, FP8_OUTPUT_TOLERANCE)
        return error

    def bind(self, hidden):
        # In-place routed scaling leaves a zero routed output at zero, so
        # repeated calls stay finite.
        routed = torch.zeros_like(hidden)
        runner, module = self.runner, self.module

        def call():
            shared, fused = runner._maybe_apply_routed_scale_to_output(module(hidden), routed)
            return shared + fused
        return call
