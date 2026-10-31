"""Captured expert routing, replayed per request (``--expert-routing-policy CAPTURED``).

``bench run --record-routed-experts`` saves vLLM's own per-request capture to
``routed_experts.npz``. Request ``bench-{i}`` -- the i-th row of the replayed
dataset -- holds ``[input_toks + output_toks - 1, layers, top_k]``: the top-k
expert IDs of every token the model processed, at every decoder layer. The last
sampled token never enters the model, hence the ``- 1``.

The simulator numbers a flat dataset's requests in the same row order, so its
request ``i`` replays exactly those rows. At each MoE layer of a step, the
routed tokens are the positions that step schedules for each request
(``Batch.routing_segments``), and under DP+EP every member's. Routing depends
on the tokens and the weights rather than on the deployment or the batching,
up to numerical differences between kernels, so one capture can serve other
TP/EP/DP layouts of the same model and workload. Padding rows are never
captured; see ``GateRouter._captured_route_ep`` for what that charges.
"""
from pathlib import Path

import numpy as np

FILE = "routed_experts.npz"

# One load per run. A gate is rebuilt for every batch, so it must not reread.
_CACHE = {}


def load(path, config, top_k):
    """The capture at ``path`` (a bench run directory or the .npz), checked
    against this model's layer count, expert count and top-k."""
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / FILE
    resolved = resolved.resolve()
    key = (str(resolved), int(config["num_hidden_layers"]), int(top_k))
    if key not in _CACHE:
        _CACHE[key] = CapturedRouting(resolved, config, top_k)
    return _CACHE[key]


class CapturedRouting:
    def __init__(self, path, config, top_k):
        from .utils import num_experts
        if not path.is_file():
            raise FileNotFoundError(
                f"No routed-experts capture at {path}; record one with "
                "`bench run --record-routed-experts`")
        self.path = path
        self.layers = int(config["num_hidden_layers"])
        self.top_k = int(top_k)
        self.num_experts = int(num_experts(config))
        with np.load(path) as archive:
            self.arrays = {name: archive[name] for name in archive.files}
        if not self.arrays:
            raise ValueError(f"{path} holds no requests")
        for name, array in self.arrays.items():
            if array.ndim != 3 or array.shape[1:] != (self.layers, self.top_k):
                raise ValueError(
                    f"{path}: {name} has shape {array.shape}; this model needs "
                    f"[rows, {self.layers}, {self.top_k}]")
            if array.dtype.kind != "u" or (array.size and int(array.max()) >= self.num_experts):
                raise ValueError(
                    f"{path}: {name} holds IDs outside this model's {self.num_experts} experts")

    def check_requests(self, requests):
        """Refuse a dataset whose requests are not the captured ones.

        ``requests`` are the router's loaded rows: ``index`` is the dataset
        row, ``output_toks`` the whole sequence length (prompt + output).
        """
        problems = []
        for req in requests:
            name = f"bench-{req['index']}"
            array = self.arrays.get(name)
            rows = int(req["output_toks"]) - 1
            if array is None:
                problems.append(f"{name} is missing")
            elif array.shape[0] != rows:
                problems.append(f"{name} has {array.shape[0]} rows, the request processes {rows}")
            if len(problems) == 5:
                break
        if problems:
            raise ValueError(
                f"{self.path} does not capture this dataset's requests: "
                + "; ".join(problems)
                + ". Replay the dataset and --num-reqs the bench run used.")

    def rows(self, layer, segments):
        """Top-k IDs ``[tokens, top_k]`` of the positions ``segments`` schedule.

        ``segments`` are ``(request_id, start, end)`` token ranges. A DP dummy
        step has none: its tokens carry no captured routing.
        """
        parts = []
        for request_id, start, end in segments:
            array = self.arrays[f"bench-{request_id}"]
            if not 0 <= start <= end <= array.shape[0]:
                raise ValueError(
                    f"bench-{request_id}: positions [{start}, {end}) exceed its "
                    f"{array.shape[0]} captured rows")
            parts.append(array[start:end, layer])
        if not parts:
            return np.empty((0, self.top_k), dtype=np.int64)
        return np.concatenate(parts).astype(np.int64)
