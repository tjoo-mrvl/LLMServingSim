"""MoE forced-routing hook.

To profile the MoE block cleanly across the (tokens, activated_experts)
grid we need to control which experts receive tokens — relying on the
(dummy-weighted) learned gate would give unpredictable activation
patterns and poor grid coverage.

This module provides:

* ``ExpertRoute.forge``: build the ``(topk_weights, topk_ids)`` tensors
  that would force a given number of experts to be activated over a
  given number of tokens.

* ``force_moe_routing``: a context manager that monkey-patches the
  active FusedMoE forward method (``forward_native`` on vLLM ≤
  v0.19.x, ``forward`` on vLLM ≥ v0.20.0 after the
  ``forward_native``/``forward_cuda`` consolidation) for the duration
  of the block so that ``self.router.select_experts`` returns our
  forged tensors instead of whatever the actual learned gate produces.
  The method name is picked at runtime via ``hasattr``. The patch is
  reverted on exit.

The patched symbols are internal vLLM APIs: the FusedMoE forward and
``self.router.select_experts`` / ``self.router._compute_routing``. A
vLLM bump that goes further than the v0.20 forward consolidation
(e.g., a router class refactor that drops ``_compute_routing``) will
require updating ``force_moe_routing``. The hook raises a clear
``RuntimeError`` when the router doesn't expose the expected internals,
pointing at where to look.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Iterator

import torch


@dataclass
class ExpertRoute:
    """Precomputed tensors that pin routing to a specific expert set.

    Attributes:
        layer_name: Identifier of the FusedMoE layer this route targets.
            The runtime patch only applies when the currently-forwarding
            layer's ``layer_name`` matches.
        weights: Tensor of shape (num_tokens, top_k); each row is a
            uniform ``1/top_k`` distribution over the chosen experts.
        ids: Integer tensor of shape (num_tokens, top_k) naming the
            experts each token is routed to.
    """

    layer_name: str
    weights: torch.Tensor
    ids: torch.Tensor

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def forge(
        cls,
        layer,
        num_tokens: int,
        activated_experts: int,
    ) -> "ExpertRoute":
        """Allocate ``weights`` and ``ids`` for a single FusedMoE layer.

        Args:
            layer: The ``FusedMoE`` instance we'll patch. Used to read
                ``top_k`` and the expected indices dtype.
            num_tokens: Number of tokens being routed this call.
            activated_experts: How many distinct experts should receive
                at least one token. Must satisfy
                ``top_k <= activated_experts <= num_tokens * top_k``.
        """
        top_k = layer.top_k
        if activated_experts < top_k:
            raise ValueError(
                f"activated_experts ({activated_experts}) must be >= "
                f"top_k ({top_k})"
            )
        if activated_experts > num_tokens * top_k:
            raise ValueError(
                f"activated_experts ({activated_experts}) cannot exceed "
                f"num_tokens*top_k ({num_tokens * top_k})"
            )

        ids_rows = _cycle_expert_ids(num_tokens, top_k, activated_experts)

        # vLLM's router cares about the dtype of topk_ids: some kernels
        # expect int32, others a specific dtype reported by the router.
        indices_dtype = layer.router._get_indices_type()
        device = next(layer.parameters()).device

        ids = torch.tensor(
            ids_rows,
            device=device,
            dtype=torch.int32 if indices_dtype is None else indices_dtype,
        )
        # Shape sanity check — the kernel will complain later with a
        # cryptic message if this is wrong, so we prefer to fail early.
        expected_shape = (num_tokens, top_k)
        if tuple(ids.shape) != expected_shape:
            raise ValueError(
                f"Forged topk_ids shape mismatch: expected {expected_shape}, "
                f"got {tuple(ids.shape)}"
            )

        weights = torch.full(
            (num_tokens, top_k),
            1.0 / top_k,
            device=device,
            dtype=torch.float32,
        )
        return cls(
            layer_name=layer.layer_name,
            weights=weights,
            ids=ids,
        )


def _cycle_expert_ids(
    num_tokens: int,
    top_k: int,
    activated_experts: int,
) -> list[list[int]]:
    """Assign expert ids deterministically so exactly ``activated_experts``
    distinct ids appear, cycled across the token dimension.

    The specific assignment doesn't matter for latency — only the count
    of distinct activations does. We use the simplest pattern:
    ``id = (token_idx * top_k + offset) % activated_experts``.
    """
    return [
        [
            (token_idx * top_k + offset) % activated_experts
            for offset in range(top_k)
        ]
        for token_idx in range(num_tokens)
    ]


# ---------------------------------------------------------------------------
# Context manager: live FusedMoE patch
# ---------------------------------------------------------------------------

@contextmanager
def force_moe_routing(route: ExpertRoute | None) -> Iterator[None]:
    """Patch ``FusedMoE``'s forward method to use ``route`` when called.

    Targets ``forward_native`` on vLLM ≤ v0.19.x and the consolidated
    ``forward`` on vLLM ≥ v0.20.0 (which collapsed
    ``forward_native``/``forward_cuda`` into a single dispatch). The
    name is picked at runtime via ``hasattr`` so the hook supports both
    eras without a hard version pin. The wrapped signature uses
    ``*args, **kwargs`` to pass through whichever positional/keyword
    arguments the active vLLM version expects (v0.20.x added an
    ``input_ids`` parameter).

    If ``route`` is None the function is a no-op (useful for dense
    profile categories where we still pass through the MoE-aware
    execute path).

    The patch is layer-scoped: only the specific FusedMoE whose
    ``layer_name`` matches ``route.layer_name`` is affected; other MoE
    layers (if any) fall through to their normal forward. This matters
    if the model has multiple MoE layers and we're profiling only one
    at a time. For the single-layer test model we override to 1
    decoder layer, this is moot.
    """
    if route is None:
        yield
        return

    # Local import so that host-side code doesn't pay the vLLM import
    # cost just to read this module.
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    # Pick the right method name for the active vLLM version.
    # v0.19.0 and earlier: forward_native (+ forward_cuda delegates to it).
    # v0.20.0+: forward (the two variants were consolidated; runner.forward
    # is dispatched to internally).
    if hasattr(FusedMoE, 'forward_native'):
        target_method = 'forward_native'
    else:
        target_method = 'forward'
    original_forward = getattr(FusedMoE, target_method)

    @wraps(original_forward)
    def hooked_forward(self, hidden_states, router_logits, *args, **kwargs):
        # Only patch the specific layer we care about. Any other
        # FusedMoE encountered during this forward pass uses its
        # normal routing.
        if self.layer_name != route.layer_name:
            return original_forward(self, hidden_states, router_logits, *args, **kwargs)

        # We also sanity-check that our forged topk_ids matches the
        # actual per-call token count. If hidden_states is padded
        # differently than expected, bail.
        expected_shape = (hidden_states.shape[0], self.top_k)
        if tuple(route.ids.shape) != expected_shape:
            raise ValueError(
                f"Forged topk_ids shape mismatch for {self.layer_name}: "
                f"expected {expected_shape}, got {tuple(route.ids.shape)}"
            )

        # The router's public API is select_experts(...). We override
        # the underlying _compute_routing to return our forged pair,
        # restore it after the single call completes. This is cleaner
        # than swapping select_experts wholesale because
        # select_experts does normalization / validation around
        # _compute_routing that we still want to run.
        if not hasattr(self.router, 'select_experts') or not hasattr(self.router, '_compute_routing'):
            raise RuntimeError(
                f"MoE forced-routing patch requires "
                f"self.router.select_experts and self.router._compute_routing "
                f"on the FusedMoE layer's router (got {type(self.router).__name__}). "
                f"vLLM may have refactored the router API — update moe_hook.py "
                f"to match the new internals. See "
                f"vllm/model_executor/layers/fused_moe/router/ for the current "
                f"router class hierarchy."
            )
        original_select_experts = self.router.select_experts
        original_compute_routing = self.router._compute_routing

        @wraps(original_select_experts)
        def hooked_select_experts(*sargs, **skwargs):
            @wraps(original_compute_routing)
            def forced_compute_routing(*cargs, **ckwargs):
                # Args are deliberately ignored — the whole point of
                # forced routing is that we return pre-forged values
                # regardless of the learned gate's logits. Signature
                # accepts whatever the active router version passes.
                return route.weights, route.ids

            self.router._compute_routing = forced_compute_routing
            try:
                return original_select_experts(*sargs, **skwargs)
            finally:
                # Always restore _compute_routing, even if select_experts
                # raises (otherwise subsequent MoE calls in the same
                # profile session would keep returning our forged values).
                self.router._compute_routing = original_compute_routing

        self.router.select_experts = hooked_select_experts
        try:
            return original_forward(self, hidden_states, router_logits, *args, **kwargs)
        finally:
            self.router.select_experts = original_select_experts

    setattr(FusedMoE, target_method, hooked_forward)
    try:
        yield
    finally:
        # Restore the original class method so future calls (in tests,
        # or after this profile session ends) behave normally.
        setattr(FusedMoE, target_method, original_forward)


# ---------------------------------------------------------------------------
# Helpers that run worker-side
# ---------------------------------------------------------------------------

def single_moe_layer(model_runner):
    """Return the model's lone ``FusedMoE`` layer.

    We run profiling with ``hf_overrides.num_hidden_layers=1`` so
    there's exactly one of every decoder sub-module — including MoE.
    If for some reason there are zero or more-than-one, raise so the
    caller can investigate rather than forge the wrong route.
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    model = model_runner.get_model()
    moe_layers = [m for m in model.modules() if isinstance(m, FusedMoE)]
    if len(moe_layers) != 1:
        raise RuntimeError(
            f"Expected exactly one FusedMoE layer in the test model, "
            f"got {len(moe_layers)}"
        )
    return moe_layers[0]
