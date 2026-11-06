"""Size the attention and skew engines' KV pool to what their plan can use, and say what an OOM means.

vLLM gives the KV pool every byte its utilization leaves after the weights and the activation peak of a prefill
with no cached context. The attention and skew plans use a fraction of that: the largest shot they can build is
``max_num_seqs`` requests of at most one context length each. What the pool does not use is exactly the room the
temporaries of a shot that carries history need (MLA's context decompression, a wide decode batch's split
buffers), which that startup budget never saw. Pinning the pool to the plan's own bound hands the rest back.

The bound comes from the limits alone, so it is known before any shot and needs no plan enumeration:
``max_num_seqs * align(L)``, with ``L`` the longest single request any shot of either plan can hold. It is an
upper bound on every shot's block-aligned KV, so against a pinned pool the planners' KV-budget filters pass every
shot they would have passed against the natural one. The plan, its resume keys and the recorded engine limits all
come from the natural boot and do not change; only the engine that fires the shots is rebooted.
"""
from . import logger as log

MAX_PIN_ATTEMPTS = 3


def request_length_bound(args, limits):
    """The longest single request (new tokens plus history) a shot of the attention or skew plan can hold.

    A decode request is ``q + kv`` with ``kv <= attention_max_kv``; a prefill request is its chunk plus context,
    and both are capped by the key axis' top (``attention_max_kv`` plus half a token budget) and by the engine's
    context. Skew requests are ``query + history`` with ``history <= attention_max_kv`` and ``sum(query)``
    at most the token budget. Every case is below ``attention_max_kv + max(token budget, query length)``.
    """
    assert args.attention_max_kv is not None, "resolve_attention_max_kv before sizing the pool"
    q_max = max((int(q) for q in args.attention_decode_q_lens), default=1)
    return max(1, min(limits.max_model_len - 1,
                      args.attention_max_kv + max(limits.max_num_batched_tokens, q_max)))


def pool_tokens_bound(args, limits):
    """Block-aligned KV tokens no shot of the attention or skew plan can exceed."""
    length = request_length_bound(args, limits)
    block = limits.block_size
    return limits.max_num_seqs * (-(-length // block) * block)


def pin_kv_pool(llm, tmpdir, args, limits, respawn, label):
    """Return ``(llm, tmpdir)``: this engine, or one rebooted with its KV pool pinned to the plan's bound.

    ``limits`` are the natural boot's and stay the planning limits. The pool is pinned in blocks
    (``num_gpu_blocks_override``), which vLLM maps to bytes for any cache-group layout. Its size is the largest of

    * what the worker's own page accounting needs for the bound shot (every group, so a hybrid model's recurrent
      state pages count; the token capacity does not see them),
    * one maximum-length request, which vLLM requires of any pool, and
    * enough group-aware tokens for the planners' filters to pass the bound (the unit they compare against).

    A pool that would not be smaller keeps the engine as it is: the plan then fills it and there is nothing to
    hand back. Every outcome is checked on the booted engine, and a pool that falls short is grown and rebooted
    rather than run. ``respawn(overrides)`` boots an engine like ``llm``'s with the private overrides.
    """
    from .engine import kv_pool_size, spin_down

    try:
        bound_tokens = pool_tokens_bound(args, limits)
        sizes = llm.collective_rpc('kv_pool_requirement', args=(
            limits.max_num_seqs, request_length_bound(args, limits), limits.max_model_len))[0]
        natural_blocks, natural_tokens = kv_pool_size(llm)
        free_natural = _free_gib(llm)
        blocks = max(sizes['bound'], sizes['context'],
                     -(-natural_blocks * bound_tokens // natural_tokens) + 1)
        if blocks >= natural_blocks:
            log.info("%s KV pool: %d blocks, no larger than the %d the plan can use; not pinned",
                     label, natural_blocks, blocks)
            llm._profiler_kv_pool = dict(pinned=False)
            return llm, tmpdir
        for attempt in range(MAX_PIN_ATTEMPTS):
            spin_down(llm, tmpdir)
            llm = tmpdir = None
            with log.stage(f"{label}  rebooting vLLM engine with a {blocks}-block KV pool"):
                llm, _, tmpdir = respawn({'num_gpu_blocks_override': blocks})
            got_blocks, got_tokens = kv_pool_size(llm)
            cache = llm.llm_engine.vllm_config
            if (cache.cache_config.block_size != limits.block_size
                    or cache.model_config.max_model_len != limits.max_model_len):
                raise ValueError("The pinned engine resolved a different block size or context length "
                                 "than the natural one; the planning limits would not describe it")
            now = llm.collective_rpc('kv_pool_requirement', args=(
                limits.max_num_seqs, request_length_bound(args, limits), limits.max_model_len))[0]
            short = max(now['bound'], now['context']) > got_blocks or got_tokens < bound_tokens
            if not short:
                llm._profiler_kv_pool = dict(pinned=True, blocks=got_blocks, tokens=got_tokens)
                log.info("%s KV pool pinned to what the plan can use: %d -> %d blocks, %d -> %d group-aware "
                         "tokens (plan bound %d); GPU memory free after boot %.2f -> %.2f GiB",
                         label, natural_blocks, got_blocks, natural_tokens, got_tokens, bound_tokens,
                         free_natural, _free_gib(llm))
                return llm, tmpdir
            blocks = max(blocks + 1, now['bound'], now['context'],
                         -(-got_blocks * bound_tokens // got_tokens) + 1 if got_tokens < bound_tokens else 0)
        raise ValueError(f"Could not size the KV pool for the plan in {MAX_PIN_ATTEMPTS} boots: "
                         f"{got_blocks} blocks hold {got_tokens} tokens against a bound of {bound_tokens}")
    except BaseException:
        if llm is not None:
            spin_down(llm, tmpdir)
        raise


def _free_gib(llm):
    return llm.collective_rpc('memory_snapshot', args=(False,))[0]['free'] / 2**30


def is_out_of_memory(exc):
    return 'out of memory' in str(exc).lower()


def explain_out_of_memory(exc, *, llm, label, done, total):
    """Say what an OOM in a sweep means and how to continue. The caller re-raises."""
    if getattr(llm, '_profiler_kv_pool', {}).get('pinned'):
        remedy = ("The KV pool is already pinned to what the plan can use, so --gpu-memory-utilization frees "
                  "nothing more. Lower --max-num-batched-tokens, --max-num-seqs or --attention-max-kv (the "
                  "plan and its temporaries shrink together), or use a GPU with more memory.")
    else:
        remedy = ("This engine's KV pool is no larger than the plan's largest shot, so the plan's own memory "
                  "fills it. --gpu-memory-utilization sets how much of the GPU the pool takes and so how much "
                  "is left for a shot's temporaries, but a smaller pool also drops the shots that no longer "
                  "fit; lowering --max-num-batched-tokens, --max-num-seqs or --attention-max-kv shrinks the "
                  "plan instead.")
    log.error("%s: CUDA out of memory at shot %d of %d (%s). %s Completed shots are kept; rerun the same "
              "command to resume.", label, done, total, str(exc).splitlines()[0][:160], remedy)
