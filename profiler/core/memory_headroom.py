"""Fire the shots that need the most temporary GPU memory first, and say how much room they left.

vLLM sizes a worker's memory at startup from a prefill that has no cached context. A shot that carries history
(for MLA, the context is decompressed into temporaries; a wide decode batch allocates split buffers) needs memory
that profiling never saw, and the shot that does not fit can sit hours into a sweep. The plan stays the same set
of shots; only their order changes, so resume keys, checkpoints and the published tables do not.

The ordering is generic. It never reads a model family: three proxies rank every shot, and a handful of the
heaviest under each fire first. The headroom report is a measurement of those shots, not a model of them.
"""
import heapq

PROBES_PER_PROXY = 4
# Every shot starts from the resident state (the worker releases its cache after each one), so the heaviest shots
# bound the rest. The proxies only rank shots, so the heaviest may not be the greatest: stop when less than the
# floor, or a tenth of what they used, remains beyond it. This is a floor against a near miss, not a prediction.
# Heavier shots outside the probe were 3-4 times lighter in every measured case.
THIN_MARGIN_FLOOR = 512 * 2**20
THIN_MARGIN_RELATIVE = 0.10


class HeadroomError(RuntimeError):
    """The heaviest shots left too little free memory for the rest of the sweep to be trusted."""


def _prefill_of(shot):
    """The requests that are prefill chunks (the same rule as ``categories._attn_key``)."""
    requests = list(shot.requests)
    q = max(1, getattr(shot, 'decode_q_len', 1))
    count = int(getattr(shot, 'n_prefill', 0) or 0)
    if not count:
        count = 1 if (requests and requests[0][0] > q) else 0
    return requests[:count]


def proxies(shot):
    """Three sizes a shot's temporaries grow with: ``(prefill tokens + their history, new tokens, KV tokens)``.

    Requests are ``(new_tokens, history)``. A prefill that carries history materialises it, a wide decode
    batch is wide in new tokens, and every request's attention reads its whole cache.
    """
    requests = list(shot.requests)
    prefill = _prefill_of(shot)
    new = sum(n for n, _ in requests)
    history = sum(h for _, h in requests)
    return (sum(n + h for n, h in prefill), new, new + history)


def heaviest(items, shot_of, *, skip=(), key_of=None, per_proxy=PROBES_PER_PROXY):
    """The distinct items heaviest under each proxy, interleaved from the heaviest down.

    One pass over ``items`` keeping ``3 * per_proxy`` of them, so a streamed plan is never materialised. Ties
    keep plan order. ``skip`` holds keys that need no firing (already measured).
    """
    heaps = [[], [], []]
    for index, item in enumerate(items):
        if key_of is not None and key_of(item) in skip:
            continue
        sizes = proxies(shot_of(item))
        for rank, heap in enumerate(heaps):
            score = (sizes[rank], sizes[(rank + 1) % 3], sizes[(rank + 2) % 3])
            entry = (score, -index, item)
            if len(heap) < per_proxy:
                heapq.heappush(heap, entry)
            elif entry[:2] > heap[0][:2]:
                heapq.heapreplace(heap, entry)
    ordered = [sorted(heap, key=lambda entry: entry[:2], reverse=True) for heap in heaps]
    chosen, seen = [], set()
    for depth in range(per_proxy):
        for column in ordered:
            if depth < len(column) and -column[depth][1] not in seen:
                seen.add(-column[depth][1])
                chosen.append(column[depth][2])
    return chosen


def is_out_of_memory(exc):
    return 'out of memory' in str(exc).lower()


def explain_out_of_memory(log, exc, *, label, utilization, done, total):
    """Say what an OOM in a sweep means and how to continue. The caller re-raises."""
    log.error("%s: CUDA out of memory at shot %d of %d (%s). The shots most likely to need temporary memory "
              "fire first, so a failure here means even the heaviest of the plan did not fit the room "
              "--gpu-memory-utilization %.2f leaves. The utilization is not part of the measurement "
              "identity: rerun the same command with a lower value (try %.2f); completed shots are kept.",
              label, done, total, str(exc).splitlines()[0][:160], utilization, max(0.5, utilization - 0.1))


class Headroom:
    """Worker memory before and after the heaviest shots, from the allocator and the driver."""

    def __init__(self, llm, label, log, utilization):
        self.llm, self.label, self.log, self.utilization = llm, label, log, utilization
        self.before = None

    def _snapshot(self, reset_peak=False):
        snaps = self.llm.collective_rpc('memory_snapshot', args=(reset_peak,))
        return min(snaps, key=lambda snap: snap['free'])

    def begin(self):
        self.before = self._snapshot(reset_peak=True)

    def finish(self, probes):
        """Report the headroom; raise ``HeadroomError`` when it is too thin to trust the rest of the sweep."""
        after = self._snapshot()
        gib = lambda value: value / 2**30
        used = max(0, after['peak_reserved'] - self.before['reserved'])
        spare = self.before['free'] - used
        required = max(THIN_MARGIN_FLOOR, THIN_MARGIN_RELATIVE * used)
        self.log.info("%s memory headroom: the %d heaviest shots (largest history, new tokens and KV) fired "
                      "first. Each took at most %.2f GiB beyond the %.2f GiB resident (peak allocated +%.2f "
                      "GiB) from the %.2f of %.2f GiB free, leaving %.2f GiB; %.2f GiB is required.",
                      self.label, probes, gib(used), gib(self.before['reserved']),
                      gib(after['peak_allocated'] - self.before['allocated']), gib(self.before['free']),
                      gib(after['total']), gib(spare), gib(required))
        if spare < required:
            suggested = max(0.5, int(100 * (self.utilization - (required - spare) / after['total']
                                            - 0.02)) / 100)
            raise HeadroomError(
                f"{self.label}: the heaviest shots used {gib(used):.2f} of the {gib(self.before['free']):.2f} "
                f"GiB free, leaving {gib(spare):.2f} GiB, and {gib(required):.2f} GiB is required (at "
                "least a tenth of what they used, for a shot the proxies rank lower). Rerun the same command with "
                f"--gpu-memory-utilization {suggested:.2f} (now {self.utilization:.2f}); the utilization is "
                "not part of the measurement identity and completed shots are kept.")
        return dict(used=used, spare=spare, free_before=self.before['free'], total=after['total'])
