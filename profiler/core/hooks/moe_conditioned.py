"""Measure complete native expert reuse cycles, timed once per point."""
from collections import Counter
import hashlib
from pathlib import Path

import torch

from ..moe_conditioning import expert_conditioning
from .eager_measurement import extract_eager_forwards
from .moe_expert_region import NativeExpertRegion
from .skew_measurement import MARKER


def expert_memory_need(owner, n, active, iterations):
    """What conditioning a point with ``n`` rows and ``active`` experts must find free.

    The guard includes one extra conversion-sized buffer and an independent
    output reference. It is evaluated from the same numbers by the preflight
    and by every point, and is never relaxed after an OOM.
    """
    plan = expert_conditioning(owner.contract, active, iterations)
    banks = plan['banks']
    additional = max(0, banks-len(owner.expert_banks))
    reserve = max(2**30, 8*n*owner.cfg.experts_per_token*owner.cfg.hidden_dim*4)
    live = owner.contract['expert_region']['schema'] == 'native-live-expert-region'
    bank_bytes = owner.contract['expert_bank_bytes'] if live else owner.contract['expert_weight_bytes']
    # A live bank clones already processed native tensors once, without a
    # second conversion buffer. Account all persistent scales in its footprint
    # and recheck free memory before each bank. The legacy conversion path
    # keeps its original double-buffer and quarter-device guards.
    needed = additional*bank_bytes*(1 if live else 2)+reserve
    return dict(plan=plan, banks=banks, additional=additional, reserve=reserve, live=live,
                bank_bytes=bank_bytes, needed=needed)


def _refusal(need, n, free, total, owner):
    gib = lambda value: f'{value/2**30:.2f} GiB'
    return MemoryError(
        f'Native expert conditioning at {n} rows needs {gib(need["needed"])} free '
        f'({need["additional"]} more weight bank(s) of {gib(need["bank_bytes"])} and a '
        f'{gib(need["reserve"])} scratch reserve, which grows with the row count) but {gib(free)} of '
        f'{gib(total)} is free with {len(owner.expert_banks)} bank(s) resident. '
        '--gpu-memory-utilization does not change this: the component engine fixes its KV cache '
        'at an explicit size. Lower --max-num-batched-tokens (the reserve scales with it) or use '
        'a GPU with more memory; the guard is never relaxed.')


def preflight_experts(owner, points, iterations):
    """Check the most demanding expert point before any is timed.

    ``points`` are ``(rows, active_experts)`` pairs. A point that cannot fit
    would otherwise fail only when the sweep reaches it, after every smaller
    point has been spent. Nothing is allocated here.
    """
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(owner.device)
    worst = None
    for n, active in points:
        need = expert_memory_need(owner, n, active, iterations)
        quarter = (not need['live'] and need['additional']*owner.contract['expert_weight_bytes'] > total//4)
        if worst is None or need['needed'] > worst[1]['needed'] or quarter:
            worst = (n, need, quarter)
        if quarter:
            break
    if worst is None:
        return dict(checked=0, free=free, total=total)
    n, need, quarter = worst
    if need['needed'] > free or quarter:
        raise _refusal(need, n, free, total, owner)
    return dict(checked=len(points), rows=n, needed=need['needed'], free=free, total=total,
                banks=need['banks'], bank_bytes=need['bank_bytes'], reserve=need['reserve'])


@torch.inference_mode()
def measure_experts(owner, point, iterations, failure_dir=None):
    if point['mode'] != 'eager':
        raise ValueError('Native MoE component acquisition requires eager mode')

    from .moe_components import ProfileCall

    n, active, pairs = (point[k] for k in ('tokens','activated_experts','local_assignments'))
    need = expert_memory_need(owner, n, active, iterations)
    plan, banks, additional, reserve = need['plan'], need['banks'], need['additional'], need['reserve']
    live, bank_bytes = need['live'], need['bank_bytes']
    free, total = torch.cuda.mem_get_info(owner.device)
    if need['needed'] > free or (not live and additional*owner.contract['expert_weight_bytes'] > total//4):
        raise _refusal(need, n, free, total, owner)
    for _ in range(additional):
        if live:
            free, _ = torch.cuda.mem_get_info(owner.device)
            if bank_bytes+reserve > free:
                raise MemoryError(
                    f'Native weight bank ({bank_bytes/2**30:.2f} GiB) plus the {reserve/2**30:.2f} GiB scratch '
                    f'reserve needs more than the {free/2**30:.2f} GiB free. --gpu-memory-utilization does not '
                    'change this (the component engine fixes its KV cache): lower --max-num-batched-tokens, '
                    'which the reserve scales with, or use a GPU with more memory.')
            owner.expert_banks.append(owner.expert.clone())
        else:
            # Every bank is rebuilt from its own copy of the checkpoint, scales
            # included (moe_quant.ExpertWeights), through the same format.
            owner.expert_banks.append(NativeExpertRegion(
                owner.cfg, owner.checkpoint_weights.clone(), owner.format))
    generator = torch.Generator(device=owner.device).manual_seed(n)
    hidden = torch.randn((n,owner.cfg.hidden_dim),device=owner.device,
                         dtype=owner.cfg.in_dtype,generator=generator)
    routes = owner.place.balanced_route(n,active,pairs)
    base_ids = torch.tensor(routes,device=owner.device,dtype=owner.indices_dtype)
    local = torch.tensor(owner.place.expert_map,device=owner.device)[base_ids.long()] >= 0
    first = owner.place.local_global_ids[0]
    ids = [torch.where(local,(base_ids-first+offset)%owner.cfg.num_local_experts+first,base_ids)
           for offset in plan['offsets']]
    weights = torch.full(base_ids.shape,1/owner.cfg.experts_per_token,device=owner.device,dtype=torch.float32)
    histogram = Counter(e for r in routes for e in r if owner.place.expert_map[e]>=0)
    if len(histogram)!=active or sum(histogram.values())!=pairs:
        raise ValueError('Conditioned routing changed the requested local work')
    # NoDPEP writes the already-reduced expert result directly into the
    # caller's output. A distinct final buffer would add a non-native copy.
    output_alias = torch.empty_like(hidden) if owner.target.dp == 1 and not live else None
    calls = [[ProfileCall(bank.bind(hidden,route,weights,output_alias=output_alias)) for route in ids]
             for bank in owner.expert_banks[:banks]]
    for bank_calls in calls:
        for r,call in enumerate(bank_calls):
            # Clone before another bank reuses the native output workspace.
            reference = calls[0][r]().clone()
            for _ in range(3): result=call()
            if output_alias is not None and result is not output_alias:
                raise ValueError('Native DP1 expert did not preserve its output alias')
            if not torch.isfinite(reference).all().item():
                raise ValueError('Non-finite native expert reference')
            torch.testing.assert_close(result,reference,atol=0,rtol=0)
    cycle, count = plan['cycle'], plan['forwards']
    def choose(index):
        position=index%cycle
        return calls[position%banks][position//banks]
    for j in range(plan['warmup_forwards']): choose(j)()
    torch.cuda.synchronize()
    context=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
             torch.profiler.ProfilerActivity.CUDA])
    with context as hook:
        for j in range(count):
            with torch.profiler.record_function(MARKER+str(j)):
                choose(j)()
    torch.cuda.synchronize()
    try:
        raw=extract_eager_forwards(hook.profiler.kineto_results,count)
    except Exception as exc:
        if failure_dir is not None:
            import tempfile
            Path(failure_dir).mkdir(parents=True,exist_ok=True)
            path=Path(tempfile.mkdtemp(prefix='conditioned-',dir=failure_dir))/'trace.json'
            hook.export_chrome_trace(str(path))
            if hasattr(exc,'audit'):
                import json
                path.with_name('native_ownership.json').write_text(json.dumps(exc.audit,indent=2))
            raise ValueError(f'{exc}; original CUDA trace: {path}') from exc
        raise
    geometry=dict(global_top_k=owner.cfg.experts_per_token,
        local_histogram=[histogram[e] for e in owner.place.local_global_ids],
        routing_sha256=hashlib.sha256(base_ids.cpu().numpy().tobytes()).hexdigest(),
        conditioning=plan,independent_outputs_verified=True)
    return dict(raw,verified=True,geometry=geometry)
