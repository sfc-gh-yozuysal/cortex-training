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

"""Tests for the tabular frames in ``cortex_training.client.tabular``; every frame is decoded with ``wire.loads``."""

from __future__ import annotations

import pytest
import torch

from cortex_training import wire
from cortex_training.client import tabular

ROWS, CONTEXT, FEATURES = 12, 8, 5
QUERY = ROWS - CONTEXT


def call(tables=1, features=FEATURES, estimator_ids=None):
    return tabular.Call(
        x=torch.randn(ROWS, tables, features, dtype=torch.float64),
        y=torch.randint(0, 3, (CONTEXT, tables)),
        categorical_inds=[[0]] + [[] for _ in range(tables - 1)],
        estimator_ids=estimator_ids,
    )


def test_weighted_cross_entropy_frame_defaults_to_the_request_mean():
    calls = [call(), call(tables=2, features=7, estimator_ids=[3, 4])]
    targets = [torch.randint(0, 3, (QUERY, 1)), torch.randint(0, 3, (QUERY, 2))]
    frame = wire.loads(tabular.build_forward_backward_payload(calls, targets, num_classes=3))

    assert frame["processing"] == {"loss_fn": "weighted_cross_entropy", "config": {"num_classes": 3}}
    first, second = frame["microbatches"]
    assert first["kwargs"]["x"].dtype == torch.float32 and first["kwargs"]["y"].dtype == torch.float32
    assert first["kwargs"]["categorical_inds"] == [[0]]
    assert "estimator_ids" not in first["kwargs"] and second["kwargs"]["estimator_ids"] == [3, 4]
    assert second["kwargs"]["x"].shape == (ROWS, 2, 7)
    assert torch.equal(second["context"]["targets"], targets[1])
    # 12 query rows in all: every weight is 1/12, so the summed objective is the mean cross-entropy.
    for mb in frame["microbatches"]:
        assert mb["context"]["weights"].dtype == torch.float32
        torch.testing.assert_close(mb["context"]["weights"], torch.full(mb["context"]["targets"].shape, 1 / 12))


def test_explicit_weights_are_sent_as_given():
    weights = [torch.arange(QUERY, dtype=torch.float32).unsqueeze(1)]
    frame = wire.loads(
        tabular.build_forward_backward_payload([call()], [torch.zeros(QUERY, 1, dtype=torch.long)], num_classes=2,
                                               weights=weights)
    )
    torch.testing.assert_close(frame["microbatches"][0]["context"]["weights"], weights[0])


def test_output_gradient_frame_and_helper():
    outputs = [torch.randn(QUERY, 1, 999)]
    target = torch.randn(QUERY)
    loss, grads = tabular.output_gradients(outputs, lambda leaves: (leaves[0].mean(-1).squeeze(-1) - target).pow(2).mean())
    expected = 2 * (outputs[0].mean(-1).squeeze(-1) - target) / QUERY / 999
    torch.testing.assert_close(grads[0], expected.reshape(QUERY, 1, 1).expand(QUERY, 1, 999))
    frame = wire.loads(tabular.build_output_gradient_payload([call()], grads))
    assert frame["processing"] == {"loss_fn": "output_gradient"}
    assert frame["microbatches"][0]["context"]["output_grad"].shape == (QUERY, 1, 999)
    assert loss > 0


def test_forward_frame_and_result():
    frame = wire.loads(tabular.build_forward_payload([call(), call(tables=2)]))
    assert set(frame) == {"microbatches"} and [set(mb) for mb in frame["microbatches"]] == [{"kwargs"}] * 2
    outputs = tabular.outputs_from_forward_result({"outputs": [torch.zeros(QUERY, 1, 10), torch.zeros(QUERY, 2, 10)]})
    assert [o.shape for o in outputs] == [(QUERY, 1, 10), (QUERY, 2, 10)]
    with pytest.raises(RuntimeError, match="no 'outputs'"):
        tabular.outputs_from_forward_result({"logprobs": [[0.0]]})


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: tabular.Call(x=torch.zeros(ROWS, 1, FEATURES, dtype=torch.long), y=torch.zeros(CONTEXT, 1),
                              categorical_inds=[[]]), "float"),
        (lambda: tabular.Call(x=torch.zeros(ROWS, 1, FEATURES), y=torch.zeros(ROWS, 1), categorical_inds=[[]]),
         "fewer rows than x"),
        (lambda: tabular.Call(x=torch.zeros(ROWS, 2, FEATURES), y=torch.zeros(CONTEXT, 2), categorical_inds=[[]]),
         "one list per table"),
    ],
)
def test_a_malformed_call_is_rejected_before_sending(make, match):
    with pytest.raises(ValueError, match=match):
        tabular.build_forward_payload([make()])


def test_targets_must_match_each_call():
    with pytest.raises(ValueError, match="integer class indices"):
        tabular.build_forward_backward_payload([call()], [torch.zeros(QUERY + 1, 1, dtype=torch.long)], num_classes=3)
    with pytest.raises(ValueError, match="one tensor per call"):
        tabular.build_forward_backward_payload([call(), call()], [torch.zeros(QUERY, 1, dtype=torch.long)], num_classes=3)
