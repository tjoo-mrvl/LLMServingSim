"""Which attention backends and MoE kernels a worker's model holds.

Read once, after the engine has loaded, without running anything. The bench
records it beside its serving observation (``meta.json`` ``kernels``) and the
profiler records the attention part per TP (``meta.yaml``
``engine_resolved.per_tp[tp].attention_backends``), so ``bench validate`` can
warn when the benchmark ran other kernels than the profile measured. The
profile's MoE side is already in each native component contract
(``expert_region.expert_class``): the main profiling engine's own MoE kernel is
not the one a distributed target's components were timed with.
"""


def _qualname(cls):
    return f'{cls.__module__}.{cls.__qualname__}'


def snapshot(model_runner):
    """Unique (layer class, backend) pairs and MoE kernel classes, sorted."""
    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    attention = set()
    layers = get_layers_from_vllm_config(model_runner.vllm_config, AttentionLayerBase)
    for layer in layers.values():
        backend = layer.get_attn_backend()
        # A speculator marks its draft model's layers; they are kept apart so
        # a target-only comparison can leave them out.
        attention.add((_qualname(type(layer)), backend.get_name(),
                       _qualname(backend), bool(layer.is_draft_layer)))
    moe = set()
    for module in model_runner.get_model().modules():
        if not isinstance(module, MoERunner):
            continue
        method = module.routed_experts.quant_method
        kernel = method.moe_kernel
        # A method without a modular kernel applies its experts itself; its
        # class is then the only record of what ran.
        moe.add((_qualname(type(method)),
                 None if kernel is None else _qualname(type(kernel.fused_experts)),
                 None if kernel is None else _qualname(type(kernel.prepare_finalize))))
    return dict(
        attention=[dict(layer=layer, backend=name, backend_class=cls, draft=draft)
                   for layer, name, cls, draft in sorted(attention)],
        moe=[dict(quant_method=method, experts=experts, prepare_finalize=prepare)
             for method, experts, prepare in sorted(moe, key=str)])
