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

"""VAGEN fork patch: VLM-tolerant forward for trl's value-head wrapper.

Covers ``verl/models/transformers/monkey_patch.py::valuehead_forward_value_only``,
which :func:`apply_monkey_patch` installs onto
``trl.AutoModelForCausalLMWithValueHead``.

Why the patch exists (see the function docstring for detail):
  * trl unconditionally evaluates ``base_model_output.logits.float()``; several VLM
    heads return ``logits=None``, which raises AttributeError.
  * verl's value-head path only reads ``output[2]``, so materialising a
    ``(bsz, seqlen, vocab)`` logits tensor wastes GBs per micro-batch.

trl is NOT imported: the surface the patch touches (``pretrained_model``, ``v_head``,
``is_peft_model``) is small enough to stand in for, keeping this CPU-only and
independent of whether trl is installed.

Out of scope: ``apply_monkey_patch`` dispatch (needs a real PreTrainedModel) and the
FSDP engine that consumes ``output[2]``.
"""

import torch

from verl.models.transformers.monkey_patch import valuehead_forward_value_only

HIDDEN, BSZ, SEQ, VOCAB = 8, 2, 5, 16


class _BaseOutput:
    def __init__(self, hidden_state, logits, loss=None, past_key_values=None):
        # [-1] is the last layer, which is what the patch reads
        self.hidden_states = [torch.zeros_like(hidden_state), hidden_state]
        self.logits = logits
        self.loss = loss
        self.past_key_values = past_key_values


class _FakePretrainedModel:
    """Stands in for the wrapped HF model. Records the kwargs it was called with."""

    def __init__(self, hidden, logits, loss=None, past_key_values=None, peft_type=None):
        self.hidden, self._logits, self._loss, self._pkv = hidden, logits, loss, past_key_values
        self.seen_kwargs = None
        if peft_type is not None:
            self.active_peft_config = type("_Cfg", (), {"peft_type": peft_type})()

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        self.seen_kwargs = kwargs
        return _BaseOutput(self.hidden, self._logits, self._loss, self._pkv)


class _FakeValueHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.summary = torch.nn.Linear(HIDDEN, 1)

    def forward(self, x):
        return self.summary(x)


class _FakeWrapper:
    """Minimal stand-in for trl.AutoModelForCausalLMWithValueHead."""

    # installed exactly the way apply_monkey_patch installs it
    forward = valuehead_forward_value_only

    def __init__(self, logits=None, loss=None, past_key_values=None, peft_type=None):
        torch.manual_seed(0)
        self.is_peft_model = peft_type is not None
        self.v_head = _FakeValueHead()
        self.hidden = torch.randn(BSZ, SEQ, HIDDEN)
        self.pretrained_model = _FakePretrainedModel(self.hidden, logits, loss, past_key_values, peft_type)


def _ids():
    return torch.zeros(BSZ, SEQ, dtype=torch.long)


def test_tolerates_logits_none():
    """The original failure mode: a VLM head returning logits=None must not raise."""
    m = _FakeWrapper(logits=None)
    lm_logits, loss, value = m.forward(input_ids=_ids())

    assert lm_logits is None
    assert loss is None
    assert value.shape == (BSZ, SEQ)
    assert torch.isfinite(value).all()


def test_logits_dropped_even_when_present():
    """Logits are discarded on purpose rather than upcast to fp32 — verl's critic path
    never reads output[0]. Guards against a future 'helpful' revert."""
    m = _FakeWrapper(logits=torch.randn(BSZ, SEQ, VOCAB))
    lm_logits, _, value = m.forward(input_ids=_ids())

    assert lm_logits is None, "logits must be dropped; see valuehead_forward_value_only docstring"
    assert value.shape == (BSZ, SEQ)


def test_value_matches_v_head_on_last_hidden_state():
    m = _FakeWrapper()
    _, _, value = m.forward(input_ids=_ids())

    torch.testing.assert_close(value, m.v_head(m.hidden).squeeze(-1))


def test_loss_is_passed_through():
    m = _FakeWrapper(loss=torch.tensor(1.25))
    _, loss, _ = m.forward(input_ids=_ids())

    torch.testing.assert_close(loss, torch.tensor(1.25))


def test_forces_output_hidden_states():
    """hidden_states[-1] is the only thing consumed, so it must be requested."""
    m = _FakeWrapper()
    m.forward(input_ids=_ids())

    assert m.pretrained_model.seen_kwargs["output_hidden_states"] is True


def test_past_key_values_threaded_and_tuple_arity():
    sentinel = object()
    m = _FakeWrapper(past_key_values=sentinel)

    out4 = m.forward(input_ids=_ids(), return_past_key_values=True)
    assert len(out4) == 4 and out4[3] is sentinel

    out3 = m.forward(input_ids=_ids())
    assert len(out3) == 3, "trl's 3-tuple arity must be preserved for output[2] consumers"

    assert m.pretrained_model.seen_kwargs["past_key_values"] is None


def test_prefix_tuning_drops_past_key_values():
    """trl's own carve-out: PREFIX_TUNING peft models must not receive past_key_values."""
    m = _FakeWrapper(peft_type="PREFIX_TUNING")
    m.forward(input_ids=_ids(), past_key_values="should-be-dropped")

    assert "past_key_values" not in m.pretrained_model.seen_kwargs


def test_non_prefix_peft_keeps_past_key_values():
    m = _FakeWrapper(peft_type="LORA")
    m.forward(input_ids=_ids(), past_key_values="kept")

    assert m.pretrained_model.seen_kwargs["past_key_values"] == "kept"


def test_forward_requests_the_dict_output():
    """★ The patch reads ``hidden_states`` off the result, so the dict form is required
    -- and verl's fused-kernel forwards raise outright without it. Most models default
    to returning a dict, so only some families surfaced this.
    """
    from types import SimpleNamespace

    import torch

    from verl.models.transformers.monkey_patch import valuehead_forward_value_only

    seen = {}

    def fake_base(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(hidden_states=[torch.zeros(1, 2, 4)], loss=None)

    class _V:
        summary = SimpleNamespace(weight=torch.zeros(1, 4))

        def __call__(self, x):
            return torch.zeros(1, 2)

    model = SimpleNamespace(
        is_peft_model=False, pretrained_model=fake_base, v_head=_V(), _return_value=None
    )

    valuehead_forward_value_only(model, input_ids=torch.zeros(1, 2, dtype=torch.long))

    assert seen.get("return_dict") is True, f"forward did not request a dict: {sorted(seen)}"
    assert seen.get("output_hidden_states") is True
