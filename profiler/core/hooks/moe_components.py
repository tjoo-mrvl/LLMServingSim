"""Worker-local native component acquisition, with explicit token domains."""
import logging
import math

import torch

from .moe_expert_region import NativeExpertRegion
from .moe_hook import single_moe_runner
from .moe_quant import expert_format, expert_reference
from .skew_measurement import MARKER
from .eager_measurement import extract_eager_forwards
from ..moe_deployment import MoeTarget
from ..moe_geometry import ExpertPlacement
from ..metadata_names import (MOE_WEIGHT_POLICY, MOE_CONDITIONING_POLICY,
                              EAGER_ATTRIBUTION)


def _deferred_finalize_enabled(config):
    """Read vLLM's effective capability, never its request flag or method.

    Both supported APIs expose ``use_deferred_moe_finalize`` as a boolean
    property. In 0.28 ``defer_moe_finalize`` is a request flag; in 0.31 it is
    a request method, whose truthiness does not describe the running path.
    Unknown APIs are unsafe for this component boundary and fail closed.
    """
    try:
        enabled = config.use_deferred_moe_finalize
    except AttributeError as exc:
        raise NotImplementedError(
            "Native MoE components require use_deferred_moe_finalize"
        ) from exc
    if type(enabled) is not bool:
        raise TypeError("use_deferred_moe_finalize must be a boolean property")
    return enabled


class ProfileCall(torch.nn.Module):
    def __init__(self, call):
        super().__init__()
        self.call = call

    def forward(self):
        return self.call()


class ComponentMeasurement:
    def __new__(cls, model_runner, target, local_budget):
        runner = single_moe_runner(model_runner)
        # The routed experts' own quant method picks the format (moe_quant):
        # unquantized experts are rebuilt here at every EP degree, and an EP
        # rank's quantized experts too, because the engine's own kernel was
        # selected for EP1. A method no format covers, and quantized experts
        # at EP1, keep the live engine's. Likewise an EP rank's shared expert
        # is rebuilt at its TP width (moe_shared_expert); at EP1 the live
        # path's wrapper measures it.
        ep_rank = MoeTarget(**target).ep > 1
        fmt = expert_format(runner.routed_experts.quant_method, model_runner.vllm_config)
        modular = fmt is not None and (not fmt.quantized or ep_rank)
        if (not modular
                or (getattr(runner, '_shared_experts', None) is not None and not ep_rank)
                or getattr(runner, 'routed_output_transform', None) is not None
                or getattr(runner, 'routed_input_transform', None) is not None
                or runner.moe_config.skip_final_all_reduce):
            from .moe_live_components import LiveComponentMeasurement
            try:
                return LiveComponentMeasurement(model_runner, target, local_budget)
            except Exception as exc:
                import traceback
                raise RuntimeError('Native live MoE initialization failed:\n'+traceback.format_exc()) from exc
        return super().__new__(cls)

    def __init__(self, model_runner, target, local_budget):
        from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceNoOP

        self.target = MoeTarget(**target)
        self.runner = single_moe_runner(model_runner)
        runner = self.runner
        self.format = expert_format(runner.routed_experts.quant_method, model_runner.vllm_config)
        if self.format is None:
            raise NotImplementedError("Quantized MoE requires a quantization-aware component adapter")
        self.format.admit(self.target)
        cfg = runner.moe_config
        if (_deferred_finalize_enabled(cfg) or cfg.skip_final_all_reduce
                or cfg.hidden_dim != cfg.hidden_dim_unpadded
                or getattr(runner.routed_experts, 'apply_router_weight_on_input', False)):
            raise NotImplementedError("Deferred, padded-hidden or input-weighted MoE needs a separate component contract")
        self.gate = getattr(runner, 'gate', None)
        if self.gate is None:
            parents = [m for m in model_runner.get_model().modules()
                       if runner in m._modules.values() and isinstance(getattr(m, 'gate', None), torch.nn.Module)]
            if len(parents) != 1:
                raise NotImplementedError("External gate binding is absent or ambiguous")
            self.gate = parents[0].gate
        if tuple(self.gate.weight.shape) != (cfg.num_experts, cfg.hidden_dim):
            raise ValueError("The gate must retain the full global expert width")
        self.cfg, metadata = self.target.resolve(cfg, local_budget)
        self.place = ExpertPlacement(self.cfg.num_experts, self.cfg.experts_per_token,
                                     self.target.ep, self.target.rank)
        self.device = self.gate.weight.device
        gen = torch.Generator(device=self.device).manual_seed(0)
        local_e, inter, h = self.cfg.num_local_experts, self.cfg.intermediate_size_per_partition, self.cfg.hidden_dim
        w13 = torch.randn((local_e, self.cfg.w13_num_shards*inter, h), device=self.device,
                          dtype=self.cfg.in_dtype, generator=gen)*.01
        w2 = torch.randn((local_e, h, inter), device=self.device,
                         dtype=self.cfg.in_dtype, generator=gen)*.01
        # The same seeded values, stored as the checkpoint stores them. Weight
        # banks are rebuilt from these, scales included.
        self.checkpoint_weights = self.format.checkpoint(w13, w2, self.cfg.w13_num_shards)
        self.expert = NativeExpertRegion(self.cfg, self.checkpoint_weights, self.format)
        expert_weight_bytes = self.checkpoint_weights.nbytes()
        self.expert_banks = [self.expert]
        self.reduction = self.expert.experts.finalize_weight_and_reduce_impl()
        if not isinstance(self.reduction, TopKWeightAndReduceNoOP):
            raise NotImplementedError("Post-expert weight reduction needs a separately timed pre-combine region")
        if self.target.dp == 1:
            from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEPModular
            self.finalize = MoEPrepareAndFinalizeNoDPEPModular()
        else:
            self.finalize = naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPModular(self.target.sequence_parallel)
        probe_rows = min(2, self.cfg.max_num_tokens)
        hidden = torch.randn((probe_rows, h), device=self.device, dtype=self.cfg.in_dtype, generator=gen)
        with torch.inference_mode():
            weights, ids = self._gate(hidden)
        self.indices_dtype = ids.dtype
        if tuple(ids.shape) != (probe_rows, self.cfg.experts_per_token) or weights.dtype != torch.float32:
            raise ValueError("Unexpected native router output contract")
        props = torch.cuda.get_device_properties(self.device)
        self.contract = dict(metadata, schema='moe-components', expert_region=self.expert.contract,
            components=list(self.target.components), modes=['eager'],
            topk_ids_dtype=str(ids.dtype), topk_weights_dtype=str(weights.dtype),
            hidden_dim=h, intermediate_size=inter, input_dtype=str(self.cfg.in_dtype),
            gate_weight_bytes=self.gate.weight.numel()*self.gate.weight.element_size(),
            expert_weight_bytes=expert_weight_bytes,
            activation=str(self.cfg.activation),
            measurement_context=MOE_WEIGHT_POLICY,
            expert_conditioning=dict(schema=MOE_CONDITIONING_POLICY,
                l2_bytes=int(props.L2_cache_size),capacity_multiple=2,
                minimum_banks=2,
                quality_relative_tolerance=.03,representative='higher_bank_control',
                local_components='isolated_warm',warmup_complete_cycles=3),
            routing_stimulus='balanced_local_histogram_with_full_global_topk',
            eager_attribution=EAGER_ATTRIBUTION,
            expert_output_reference='independent_eager_clone',
            torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
            synthetic_expert_weights=True,
            gpu=dict(name=props.name, total_memory=props.total_memory,
                     capability=list(torch.cuda.get_device_capability(self.device))),
            excluded=['collectives', 'cpu_time', 'launch_gaps', 'shared_experts'])
        if self.format.output_tolerance is not None:
            error = self._check_expert_output(gen)
            logging.getLogger(__name__).info(
                '%s expert output check: relative error %.4f against the fp32 reference '
                '(tolerance %s)', self.format.name, error, self.format.output_tolerance)
            self.contract['expert_weight_recipe'] = self.format.recipe()
        if self.target.dp == 1:
            prepare = self._verify_local_boundary(hidden, weights, ids)
            self.contract['local_execution'] = dict(
                prepare=prepare, finalize='output_alias_noop', expert_output_alias=True)
        self.shared = None
        if getattr(runner, '_shared_experts', None) is not None:
            from .moe_shared_expert import SharedExpertCopy
            # The shared expert becomes its own component, timed on the rank's
            # local rows; component_names admits it by this contract entry, and
            # serving (serving/core/moe_execution.py) charges it after the
            # routed experts. Models without one keep their contracts unchanged.
            self.shared = SharedExpertCopy(model_runner.vllm_config, runner, self.target,
                                           self.device)
            self.shared.check_output(torch.randn((min(16, self.cfg.max_num_tokens), h),
                                                 device=self.device, dtype=self.cfg.in_dtype,
                                                 generator=gen))
            self.contract['components'].append('shared_experts')
            self.contract['excluded'].remove('shared_experts')
            self.contract['shared_experts'] = self.shared.contract

    @torch.inference_mode()
    def _check_expert_output(self, gen):
        """One kernel call against an fp32 computation of this rank's experts.

        Runs once, before any timing, over a balanced batch that activates as
        many local experts as the token count allows. A mismatch stops the
        acquisition: the seeded weights, their conversion to the selected
        backend's layout or the expert map would be wrong.
        """
        cfg = self.cfg
        n = min(16, cfg.max_num_tokens)
        hidden = torch.randn((n, cfg.hidden_dim), device=self.device, dtype=cfg.in_dtype, generator=gen)
        active = self.place.active_bounds(n)[1]
        pairs = self.place.assignment_bounds(n, active)[1]
        ids = torch.tensor(self.place.balanced_route(n, active, pairs), device=self.device,
                           dtype=self.indices_dtype)
        weights = torch.full(ids.shape, 1/cfg.experts_per_token, device=self.device, dtype=torch.float32)
        output = torch.empty_like(hidden) if self.target.dp == 1 else None
        result = self.expert.bind(hidden, ids, weights, output_alias=output)()
        reference = expert_reference(self.format, hidden, ids, weights, self.checkpoint_weights,
                                     self.place, cfg)
        error = ((result.float() - reference).norm() / reference.norm()).item()
        tolerance = self.format.output_tolerance
        if not math.isfinite(error) or error > tolerance:
            raise ValueError(f'{self.format.name} expert output differs from the fp32 reference by '
                             f'{error:.4f} (relative Frobenius norm, tolerance {tolerance})')
        return error

    @torch.inference_mode()
    def _verify_local_boundary(self, hidden, weights, ids):
        """Omit DP1 boundaries only after observing their native contract.

        Finalize must be a CUDA-free no-op on the output alias. Prepare must
        hand back the input without reordering or token metadata, and returns
        the label ``local_execution.prepare`` records: ``identity`` when it
        passes the input through -- unquantized experts, or quantized experts
        that quantize it themselves -- or ``timed_input_quantization`` when
        vLLM's prepare quantizes it (``moe_quant``: the format calls it and
        checks the result). That is work the deployment runs, so the expert
        region times it inside ``experts`` rather than omitting it.
        """
        from torch.autograd import DeviceType

        def cuda_work(hook):
            return any(event.device_type() == DeviceType.CUDA and not event.is_user_annotation()
                       for event in hook.profiler.kineto_results.events())

        def profile():
            return torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                      torch.profiler.ProfilerActivity.CUDA])

        output = torch.empty_like(hidden)
        call = self.expert.bind(hidden, ids, weights, output_alias=output)
        if call() is not output:
            raise NotImplementedError('Native DP1 expert output does not preserve its output alias')
        reference = output.clone()
        torch.cuda.synchronize()
        with profile() as prepare_hook:
            prepared, quantizes = self.format.prepare_dp1(
                self.finalize, hidden, weights, ids, self.cfg, self.expert.mapping,
                self.expert.experts)
        with profile() as finalize_hook:
            self.finalize.finalize(output, output, weights, ids, False, self.reduction)
        if len(prepared) != 5 or any(value is not None for value in prepared[2:]):
            raise NotImplementedError('Native DP1 preparation reorders tokens or adds metadata')
        if not quantizes:
            if prepared[0] is not hidden or prepared[1] is not None:
                raise NotImplementedError('Native DP1 preparation is not an unquantized identity')
            if cuda_work(prepare_hook):
                raise NotImplementedError('Native DP1 prepare/finalize performs unprofiled CUDA work')
            label = 'identity'
        elif not self.format.quantized_input(prepared, hidden):
            raise NotImplementedError(f'Native DP1 {self.format.name} preparation is not a plain '
                                      'input quantization')
        else:
            label = 'timed_input_quantization'
        if cuda_work(finalize_hook):
            raise NotImplementedError('Native DP1 prepare/finalize performs unprofiled CUDA work')
        torch.testing.assert_close(output, reference, atol=0, rtol=0)
        return label

    def _gate(self, hidden):
        logits, _ = self.gate(hidden)
        return self.runner.router.select_experts(hidden_states=hidden, router_logits=logits,
                                                  topk_indices_dtype=self.finalize.topk_indices_dtype())

    @torch.inference_mode()
    def measure(self, point, iterations, failure_dir=None):
        if point['mode'] != 'eager':
            raise ValueError('Native MoE component acquisition requires eager mode')

        from torch.profiler import record_function
        from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep

        n = point['tokens']
        if point['component'] not in self.contract['components']:
            raise ValueError('Component is not part of the native target execution')
        if type(iterations) is not int or iterations < 1:
            raise ValueError("Measurement needs positive iterations")
        if point['component'] == 'experts':
            from .moe_conditioned import measure_experts
            return measure_experts(self,point,iterations,failure_dir)
        generator = torch.Generator(device=self.device).manual_seed(n)
        hidden = torch.randn((n, self.cfg.hidden_dim), device=self.device,
                             dtype=self.cfg.in_dtype, generator=generator)
        details = {}
        original_group = naive_dp_ep.get_ep_group
        if point['component'] == 'gate_routing':
            call = lambda: self._gate(hidden)
        elif point['component'] == 'shared_experts':
            call = self.shared.bind(hidden)
        elif point['component'] == 'finalize_copy':
            output = torch.empty_like(hidden)
            gathered_n = n * (self.target.ep if self.target.sequence_parallel else self.target.dp)
            fused = torch.empty((gathered_n, self.cfg.hidden_dim), device=self.device, dtype=hidden.dtype)
            ids = torch.zeros((gathered_n, self.cfg.experts_per_token), device=self.device, dtype=self.indices_dtype)
            weights = torch.full(ids.shape, 1/self.cfg.experts_per_token, device=self.device, dtype=torch.float32)
            target = self.target
            class CompletedCombine:
                def combine(self, value, is_sequence_parallel=False):
                    if value is not fused or is_sequence_parallel != target.sequence_parallel:
                        raise ValueError("Unexpected native combine contract")
                    return hidden
            completed = CompletedCombine()
            def call():
                self.finalize.finalize(output, fused, weights, ids, False, self.reduction)
                return output
        else:
            raise ValueError("Unknown MoE component")
        try:
            if point['component'] == 'finalize_copy':
                naive_dp_ep.get_ep_group = lambda: completed
            wrapped = ProfileCall(call)
            for _ in range(3):
                result = wrapped()
            outputs = result if isinstance(result, tuple) else (result,)
            if not all(torch.isfinite(value).all().item() for value in outputs):
                raise ValueError("Non-finite native component output")
            if point['component'] == 'finalize_copy':
                torch.testing.assert_close(result, hidden, atol=0, rtol=0)
            torch.cuda.synchronize()
            context = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                       torch.profiler.ProfilerActivity.CUDA])
            with context as hook:
                for index in range(iterations):
                    with record_function(MARKER+str(index)):
                        wrapped()
            torch.cuda.synchronize()
            raw = extract_eager_forwards(hook.profiler.kineto_results, iterations)
            return dict(raw, verified=True, geometry=details)
        finally:
            naive_dp_ep.get_ep_group = original_group
