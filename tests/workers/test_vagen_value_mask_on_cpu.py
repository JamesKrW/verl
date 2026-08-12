# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional ``value_mask`` in ``workers/utils/losses.py::value_loss``.

Turn-level and bi-level advantage estimators write ``returns`` at one anchor token per
turn and leave the rest at a sentinel; ``value_mask`` marks the positions that actually
carry return supervision so the critic is not trained on sentinels. The field is
optional -- absent means "supervise every response token", i.e. upstream behaviour.

The assertions are properties rather than numeric values, so they stay valid if
``compute_value_loss`` changes:

  1. an all-ones ``value_mask`` is indistinguishable from no ``value_mask``
  2. perturbing ``returns`` where ``value_mask == 0`` does not move the loss
  3. perturbing ``returns`` where ``value_mask == 1`` does move the loss

``no_padding_2_padding`` is stubbed so the test exercises our masking logic rather than
the unpadding machinery, which has its own coverage.

Out of scope: the trainer-side plumbing that has to place ``value_mask`` into the
TrainingWorker TensorDict in the first place.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from tensordict import TensorDict

from verl.workers.utils import losses as losses_mod

BSZ, RESP = 2, 6
CONFIG = SimpleNamespace(cliprange_value=0.5, loss_agg_mode="token-mean")


def _data(returns, value_mask=None, response_mask=None) -> TensorDict:
    fields = {
        "values": torch.zeros(BSZ, RESP),
        "returns": returns,
        "response_mask": torch.ones(BSZ, RESP, dtype=torch.long) if response_mask is None else response_mask,
    }
    if value_mask is not None:
        fields["value_mask"] = value_mask
    return TensorDict(fields, batch_size=[BSZ])


def _loss(returns, value_mask=None, response_mask=None) -> float:
    """Run value_loss with a fixed vpreds so only the masking logic varies."""
    vpreds = torch.full((BSZ, RESP), 0.3)
    model_output = {"values": vpreds}
    with patch.object(losses_mod, "no_padding_2_padding", lambda t, d: vpreds):
        loss, _ = losses_mod.value_loss(CONFIG, model_output, _data(returns, value_mask, response_mask))
    return loss.item()


def _base_returns() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BSZ, RESP)


def test_absent_value_mask_is_upstream_behaviour():
    """Smoke: the unmodified path still runs and produces a finite scalar."""
    loss = _loss(_base_returns())
    assert isinstance(loss, float)
    assert loss == loss and abs(loss) != float("inf")  # not NaN, not inf


def test_all_ones_mask_equals_no_mask():
    returns = _base_returns()
    ones = torch.ones(BSZ, RESP, dtype=torch.long)

    assert _loss(returns, ones) == _loss(returns)


def test_masked_out_positions_do_not_affect_loss():
    """The core guarantee: sentinel returns behind value_mask==0 are invisible."""
    returns = _base_returns()
    mask = torch.ones(BSZ, RESP, dtype=torch.long)
    mask[:, 3:] = 0

    poisoned = returns.clone()
    poisoned[:, 3:] = -100.0  # the sentinel turn-level GAE actually writes

    assert _loss(poisoned, mask) == _loss(returns, mask)


def test_supervised_positions_still_affect_loss():
    """Counterpart to the above — the mask must not silently disable everything."""
    returns = _base_returns()
    mask = torch.ones(BSZ, RESP, dtype=torch.long)
    mask[:, 3:] = 0

    changed = returns.clone()
    changed[:, 0] += 5.0  # inside the supervised region

    assert _loss(changed, mask) != _loss(returns, mask)


def test_bool_and_int_masks_agree():
    returns = _base_returns()
    mask_int = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0]], dtype=torch.long)

    assert _loss(returns, mask_int) == _loss(returns, mask_int.bool())


def test_empty_mask_nan_is_preexisting_upstream_behaviour():
    """An empty effective mask yields NaN -- and that is upstream's behaviour, not ours.

    ``agg_loss(..., "token-mean")`` computes ``masked_sum(...) / loss_mask.sum()``; with
    an empty mask that is 0/0. Reachable today without this patch, because
    ``construct_minimal_padding_template`` gives padding rows an all-zero
    ``response_mask`` (padding_utils.py) -- so a micro-batch consisting only of padding
    rows already NaNs on unmodified verl.

    This test pins the equivalence rather than the value: if ``value_mask`` ever makes
    an empty mask reachable in a case where ``response_mask`` alone would not, the two
    branches diverge and this fails.

    Not reachable via `value_mask` on real rows in practice: turn-level GAE writes one
    anchor per turn, and a real row always has at least one turn.
    """
    returns = _base_returns()
    zeros = torch.zeros(BSZ, RESP, dtype=torch.long)
    ones = torch.ones(BSZ, RESP, dtype=torch.long)

    upstream_nan = _loss(returns, response_mask=zeros)  # no value_mask at all
    ours_nan = _loss(returns, value_mask=zeros, response_mask=ones)

    assert upstream_nan != upstream_nan, "upstream no longer NaNs on an empty mask -- re-evaluate"
    assert ours_nan != ours_nan, "value_mask path diverged from the upstream empty-mask path"


# --------------------------------------------------------------------------------
# Transport: does ``value_mask`` actually survive the trip to ``value_loss``?
#
# The tests above prove the masking maths is right *given* the key arrives. Whether it
# arrives is a separate question, and a fragile one: a whitelist anywhere on the critic
# path drops the key, after which ``response_mask & value_mask`` quietly degrades to the
# plain ``response_mask``.
#
# The current path routes through ``left_right_2_no_padding``, which mutates a
# TensorDict in place rather than selecting from it, so the change in ``value_loss`` is
# sufficient. That is an assumption about code owned elsewhere, so pin it.
# --------------------------------------------------------------------------------


def _padded_batch() -> TensorDict:
    """A left-right padded critic batch with one padding column on each side."""
    attention_mask = torch.ones(BSZ, RESP, dtype=torch.long)
    attention_mask[:, 0] = 0  # left pad
    attention_mask[:, -1] = 0  # right pad
    response_mask = torch.zeros(BSZ, RESP, dtype=torch.long)
    response_mask[:, 2:-1] = 1

    value_mask = torch.zeros(BSZ, RESP, dtype=torch.long)
    value_mask[:, 2] = 1  # one supervised anchor per row, as turn-level GAE emits

    return TensorDict(
        {
            "input_ids": torch.arange(BSZ * RESP).reshape(BSZ, RESP),
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": torch.arange(RESP).expand(BSZ, RESP).contiguous(),
            "values": torch.zeros(BSZ, RESP),
            "returns": torch.randn(BSZ, RESP),
            "value_mask": value_mask,
        },
        batch_size=[BSZ],
    )


def test_value_mask_survives_left_right_2_no_padding():
    """★ The transport guard.

    If a revision reintroduces a whitelist on the critic path, the key disappears and
    the critic silently trains on the sentinel again -- no error, and a
    *healthy-looking* ``vf_explained_var``, since a constant-dominated target is
    trivially explainable. Fail loudly here instead.
    """
    from verl.workers.utils.padding import left_right_2_no_padding

    out = left_right_2_no_padding(_padded_batch())

    assert "value_mask" in out.keys(), (
        "value_mask was dropped on the critic path; the critic will fall back to the "
        "full response_mask and regress towards the -100 sentinel"
    )


def test_value_mask_stays_dense_and_aligned_with_response_mask():
    """``value_loss`` does ``response_mask & value_mask``, so the two must keep the same
    shape and neither may be converted to a nested tensor (``input_ids`` and
    ``position_ids`` are, which is exactly why this is worth checking)."""
    from verl.workers.utils.padding import left_right_2_no_padding

    out = left_right_2_no_padding(_padded_batch())

    assert not out["value_mask"].is_nested, "value_mask must stay dense to align with response_mask"
    assert out["value_mask"].shape == out["response_mask"].shape
    # The `&` in value_loss requires an integral/bool dtype on both sides.
    assert not out["value_mask"].is_floating_point()
