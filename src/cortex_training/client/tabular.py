# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frames for tabular foundation model jobs (``model_provider: "tabular"``).

A tabular foundation model (TabPFN, NVIDIA Kumo Tabular) predicts the targets
of query rows from labeled context rows of the same table in one forward pass.
The server runs only the network; the model library's preprocessing runs in
the caller's process, and a request carries one :class:`Call` per network
call. A call is indivisible: its context and query rows stay in one forward
pass.

- ``x``: ``[R, T, F]`` float32, ``T`` tables of ``R`` rows (typically one per
  ensemble estimator); the first ``R_context`` rows are the labeled context.
- ``y``: ``[R_context, T]`` context targets (class indices for classification).
- ``categorical_inds``: one list of categorical feature indices per table.

Training uses one of two losses:

- ``weighted_cross_entropy``: ``-sum(w * log p(target))`` over the first
  ``num_classes`` outputs of every call. The server sums; the caller sets ``w``
  (the default ``1/N`` over all query rows of the request is mean
  cross-entropy).
- ``output_gradient``: the caller computes any loss ``L`` on the outputs of a
  forward pass and sends ``g = dL/doutputs``; the server trains
  ``sum(<g, outputs>)``, whose gradient is that of ``L``.

``forward`` returns each call's outputs, ``[R_query, T, n_outputs]``.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cortex_training import wire

WEIGHTED_CROSS_ENTROPY = "weighted_cross_entropy"
OUTPUT_GRADIENT = "output_gradient"
# Ask for a DSSST1 result so poll_request returns decoded tensors.
_RESPONSE_OPTIONS = {"response_options": {"format": "dssst1", "delivery": "chunked"}}


@dataclass(frozen=True)
class Call:
    """The inputs of one network call."""

    x: Any  # [R, T, F]
    y: Any  # [R_context, T]
    categorical_inds: Sequence[Sequence[int]]
    estimator_ids: Sequence[int] | None = None

    def kwargs(self) -> dict:
        import torch

        x = torch.as_tensor(self.x)
        y = torch.as_tensor(self.y)
        if not x.is_floating_point() or x.dim() != 3:
            raise ValueError(f"x must be a float [rows, tables, features] tensor, got {x.dtype} {tuple(x.shape)}")
        if y.dim() != 2 or y.shape[1] != x.shape[1] or not 0 < y.shape[0] < x.shape[0]:
            raise ValueError(f"y must be [context_rows, tables] with fewer rows than x; got x={tuple(x.shape)} y={tuple(y.shape)}")
        if len(self.categorical_inds) != x.shape[1]:
            raise ValueError(f"categorical_inds needs one list per table ({x.shape[1]}), got {len(self.categorical_inds)}")
        kwargs = {
            "x": x.to(torch.float32).contiguous(),
            "y": y.to(torch.float32).contiguous(),
            "categorical_inds": [[int(i) for i in indices] for indices in self.categorical_inds],
        }
        if self.estimator_ids is not None:
            kwargs["estimator_ids"] = [int(i) for i in self.estimator_ids]
        return kwargs

    @property
    def query_shape(self) -> tuple[int, int]:
        import torch

        x, y = torch.as_tensor(self.x), torch.as_tensor(self.y)
        return x.shape[0] - y.shape[0], x.shape[1]


def _frame(microbatches: list[dict], processing: dict | None = None) -> bytes:
    frame: dict[str, Any] = {"microbatches": microbatches}
    if processing is not None:
        frame["processing"] = processing
    return wire.dumps(frame, metadata=_RESPONSE_OPTIONS)


def build_forward_backward_payload(
    calls: Sequence[Call],
    targets: Sequence[Any],
    *,
    num_classes: int,
    weights: Sequence[Any] | None = None,
) -> bytes:
    """``weighted_cross_entropy`` frame: one ``[R_query, T]`` target tensor per call, class indices in that call's order.

    ``weights`` defaults to ``1/N`` for every query row, ``N`` the query rows of the whole request.
    """
    import torch

    if len(targets) != len(calls) or (weights is not None and len(weights) != len(calls)):
        raise ValueError("targets (and weights) need one tensor per call")
    targets = [torch.as_tensor(t) for t in targets]
    for call, target in zip(calls, targets, strict=True):
        if tuple(target.shape) != call.query_shape or target.is_floating_point():
            raise ValueError(f"targets must be integer class indices of shape {list(call.query_shape)} per call")
    if weights is None:
        total = sum(t.numel() for t in targets)
        weights = [torch.full(t.shape, 1.0 / total) for t in targets]
    microbatches = [
        {
            "kwargs": call.kwargs(),
            "context": {"targets": target.long().contiguous(), "weights": torch.as_tensor(w, dtype=torch.float32)},
        }
        for call, target, w in zip(calls, targets, weights, strict=True)
    ]
    return _frame(microbatches, {"loss_fn": WEIGHTED_CROSS_ENTROPY, "config": {"num_classes": int(num_classes)}})


def build_output_gradient_payload(calls: Sequence[Call], output_grads: Sequence[Any]) -> bytes:
    """``output_gradient`` frame: one ``dL/doutputs`` tensor per call, shaped like that call's forward outputs."""
    import torch

    if len(output_grads) != len(calls):
        raise ValueError("output_grads needs one tensor per call")
    microbatches = [
        {"kwargs": call.kwargs(), "context": {"output_grad": torch.as_tensor(g, dtype=torch.float32).contiguous()}}
        for call, g in zip(calls, output_grads, strict=True)
    ]
    return _frame(microbatches, {"loss_fn": OUTPUT_GRADIENT})


def build_forward_payload(calls: Sequence[Call]) -> bytes:
    """No-gradient ``forward`` frame; the result holds each call's outputs."""
    return _frame([{"kwargs": call.kwargs()} for call in calls])


def outputs_from_forward_result(result: Mapping[str, Any]) -> list:
    """Each call's outputs, ``[R_query, T, n_outputs]`` float32, from a polled tabular ``forward`` result."""
    import torch

    outputs = result.get("outputs")
    if outputs is None:
        raise RuntimeError(f"the forward result has no 'outputs' (keys: {sorted(result)}); is the job a tabular job?")
    return [torch.as_tensor(out, dtype=torch.float32) for out in outputs]


def output_gradients(outputs: Sequence[Any], loss_fn: Callable[[list], Any]) -> tuple[float, list]:
    """``(L, [dL/doutputs per call])`` for a loss written over the outputs of a forward pass."""
    import torch

    leaves = [torch.as_tensor(out, dtype=torch.float32).detach().clone().requires_grad_(True) for out in outputs]
    loss = loss_fn(leaves)
    if not torch.is_tensor(loss) or loss.dim() != 0:
        raise TypeError("loss_fn must return a scalar tensor")
    loss.backward()
    grads = [leaf.grad if leaf.grad is not None else torch.zeros_like(leaf) for leaf in leaves]
    if not all(torch.isfinite(g).all() for g in grads) or not torch.isfinite(loss):
        raise ValueError("the loss or its gradient is not finite")
    return float(loss.detach()), grads
