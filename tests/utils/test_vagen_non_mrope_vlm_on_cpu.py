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

"""VLMs outside the mrope family must load and run.

``hf_processor`` binds ``get_rope_index`` for the handful of models that use mrope. It
used to raise for anything else, and since that raise sits inside a broad ``except`` the
processor became ``None`` -- indistinguishable from a text-only model, and surfacing much
later as a confusing failure somewhere that expected images.

An unrecognised processor now keeps working with ordinary position ids, which is correct
for InternVL, LLaVA and similar.
"""

import ast
import inspect
import warnings

from verl.utils import tokenizer as tokenizer_mod


def _match_default_branch_source() -> str:
    """Source of the `case _:` arm of the processor dispatch."""
    tree = ast.parse(inspect.getsource(tokenizer_mod.hf_processor))
    for node in ast.walk(tree):
        if isinstance(node, ast.Match):
            for case in node.cases:
                if case.pattern.__class__.__name__ == "MatchAs" and case.pattern.pattern is None:
                    return ast.unparse(case.body)
    raise AssertionError("processor dispatch no longer has a default arm")


def test_unknown_processor_does_not_raise():
    """★ The regression. A raise here is swallowed into processor=None."""
    body = _match_default_branch_source()

    assert "raise" not in body, f"default arm still raises, which nulls the processor:\n{body}"


def test_unknown_processor_warns_about_position_ids():
    """Silently assuming ordinary position ids would be wrong for an mrope model that
    simply is not in the list yet, so the assumption has to be visible."""
    body = _match_default_branch_source()

    assert "warn" in body
    assert "mrope" in body


def test_position_ids_fall_back_without_a_rope_binding():
    """★ The other half: _compute_position_ids called get_rope_index unconditionally,
    so a processor without the binding raised AttributeError instead."""
    import torch

    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    class _PlainProcessor:
        """A VLM processor that indexes positions the ordinary way."""

    worker = AgentLoopWorker.__new__(AgentLoopWorker)
    worker.processor = _PlainProcessor()

    attention_mask = torch.tensor([[1, 1, 1, 0]])
    position_ids = worker._compute_position_ids(
        input_ids=torch.tensor([[5, 6, 7, 0]]), attention_mask=attention_mask, multi_modal_inputs={}
    )

    # Ordinary (1, seq_len) ids, not the (1, 4, seq_len) an mrope model produces.
    assert position_ids.shape == (1, 4), position_ids.shape
    # Valid positions count up; what verl writes at padding is its own convention.
    assert position_ids[0][:3].tolist() == [0, 1, 2]


def test_mrope_processors_still_get_the_binding():
    """The fallback must not swallow the mrope path -- those models need real
    vision position ids, and losing them silently would degrade training."""
    src = inspect.getsource(tokenizer_mod.hf_processor)

    for name in ("Qwen2VLProcessor", "Qwen2_5_VLProcessor", "Qwen3VLProcessor"):
        assert name in src
    assert "processor.get_rope_index = types.MethodType" in src


def test_warning_is_emitted_rather_than_printed():
    """A print would be lost in Ray worker output; a warning can be filtered and
    surfaced by the test suite."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        body = _match_default_branch_source()

    assert "warnings.warn" in body
