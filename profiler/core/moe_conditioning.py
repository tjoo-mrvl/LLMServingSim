"""Deterministic full-position weight rotation plan, independent of timings."""
import math
from .metadata_names import MOE_CONDITIONING_POLICY


def expert_conditioning(contract, active, minimum_iterations):
    local = contract['local_experts']
    if (type(local) is not int or local < 1 or type(active) is not int
            or not 0 <= active <= local or type(minimum_iterations) is not int
            or minimum_iterations < 1):
        raise ValueError('Invalid expert conditioning geometry')
    policy = contract['expert_conditioning']
    if policy['schema'] != MOE_CONDITIONING_POLICY:
        raise ValueError('Unknown expert conditioning protocol')
    if (type(policy['capacity_multiple']) is not int or policy['capacity_multiple'] < 1
            or policy['warmup_complete_cycles'] != 3):
        raise ValueError('Unsupported expert conditioning policy')
    weight_bytes = contract['expert_weight_bytes']
    l2_bytes = policy['l2_bytes']
    if any(type(v) is not int or v < 1 for v in (weight_bytes,l2_bytes)):
        raise ValueError('Measured weight size and actual GPU cache capacity are required')
    offsets = list(range(0,local,active)) if active else [0]
    banks = policy['minimum_banks']
    if type(banks) is not int or banks < 1 or banks & (banks-1):
        raise ValueError('The minimum independent weight-bank count must be a positive power of two')
    if active:
        # Rotation visits the complete local weight bank, not just the first
        # A experts. This is a conditioning guard, never a latency equation.
        while banks*weight_bytes < policy['capacity_multiple']*l2_bytes:
            banks *= 2
    else:
        banks = 1
    cycle = len(offsets)*banks
    # Whole cycles only, so every bank and offset is timed equally often.
    count = math.ceil(minimum_iterations/cycle)*cycle
    return dict(offsets=offsets,banks=banks,cycle=cycle,forwards=count,
                warmup_forwards=3*count)
