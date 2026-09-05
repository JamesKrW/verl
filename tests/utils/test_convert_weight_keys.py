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
"""`convert_weight_keys` must produce checkpoint-namespace names on any transformers.

Rollout weight sync hands these names straight to the inference engine, which looks
them up in its own ``params_dict``. A name in the wrong namespace is not reported
here -- it surfaces from inside the engine as a ``KeyError`` on a parameter that
does exist, under a different prefix.
"""

import pytest
import torch

from verl.utils.model import convert_weight_keys


class _Mapped:
    """Stands in for a transformers < 5 model: carries the mapping attribute."""

    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }


class _Unmapped:
    """A non-PreTrainedModel with no mapping: must keep the old pass-through.

    Deliberately not a PreTrainedModel. This function is on the weight-sync path and
    was previously incapable of raising; the transformers >= 5 fallback must not
    change that for callers passing a wrapper.
    """


RUNTIME_KEYS = [
    "model.visual.blocks.0.mlp.gate_proj.weight",
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "lm_head.weight",
]
CHECKPOINT_KEYS = [
    "visual.blocks.0.mlp.gate_proj.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "lm_head.weight",
]


def _state_dict(keys):
    return {key: torch.zeros(1) for key in keys}


def test_mapping_attribute_is_reversed():
    converted = convert_weight_keys(_state_dict(RUNTIME_KEYS), _Mapped())
    assert list(converted) == CHECKPOINT_KEYS


def test_non_pretrained_model_without_a_mapping_passes_through():
    keys = ["model.layers.0.mlp.up_proj.weight", "lm_head.weight"]
    assert list(convert_weight_keys(_state_dict(keys), _Unmapped())) == keys


def test_missing_mapping_does_not_silently_pass_through_a_vl_model():
    """transformers >= 5 stopped filling the mapping; the rename still has to happen.

    Skipped only where the mapping is actually populated, which is the case that
    needs no fallback. 5.8.1 keeps the attribute and sets it to `{}`, so guarding
    on the attribute alone reverses an empty mapping and silently changes nothing.
    """
    revert = pytest.importorskip(
        "transformers.core_model_loading", reason="transformers < 5 has no weight-conversion module"
    )
    if not hasattr(revert, "revert_weight_conversion"):
        pytest.skip("this transformers has no revert_weight_conversion")

    transformers = pytest.importorskip("transformers")
    from accelerate import init_empty_weights

    config = transformers.AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    with init_empty_weights():
        model = transformers.AutoModelForImageTextToText.from_config(config)
    if getattr(model, "_checkpoint_conversion_mapping", None):
        pytest.skip("this transformers still populates the mapping; the fallback is unused")

    converted = convert_weight_keys(model.state_dict(), model)
    prefixes = {key.split(".")[0] for key in converted}
    # The engine's namespace: the vision tower at the top level, the language tower
    # under `model.`. Without the fallback every key stays under `model.`.
    assert "visual" in prefixes, f"vision tower was not un-nested; got prefixes {sorted(prefixes)}"
    assert prefixes <= {"visual", "model", "lm_head"}, sorted(prefixes)
