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

"""Placeholder dedup has to hold for every VLM family, not just Qwen.

Sending prompt_ids alongside multi_modal_data requires one placeholder per item: vLLM
expands it itself from the images. An HF-tokenized prompt already carries the expanded
run, so passing it through makes vLLM expand twice.

The collapse used to be gated on ``Qwen2VLImageProcessor`` appearing in the processor's
class name, so every other family shipped the expanded run and died on a CUDA
masked_scatter assert inside the model -- an error that names neither the prompt nor the
processor.
"""

import numpy as np
import pytest

from verl.workers.rollout.utils import dedup_multimodal_placeholder_tokens as dedup


class _Processor:
    def __init__(self, image_token_id=None, video_token_id=None, class_name="LlavaProcessor"):
        if image_token_id is not None:
            self.image_token_id = image_token_id
        if video_token_id is not None:
            self.video_token_id = video_token_id
        self.__class__.__name__ = class_name


def test_run_collapses_to_one_token():
    """★ The regression: a family whose class name is not Qwen's still needs this."""
    prompt = [1, 9, 9, 9, 9, 2]

    assert dedup(prompt, _Processor(image_token_id=9)) == [1, 9, 2]


def test_separate_runs_stay_separate():
    """Two images means two placeholders; merging them would under-count features."""
    prompt = [9, 9, 1, 9, 9, 9]

    assert dedup(prompt, _Processor(image_token_id=9)) == [9, 1, 9]


def test_video_and_image_are_both_collapsed():
    prompt = [9, 9, 1, 8, 8, 8]

    assert dedup(prompt, _Processor(image_token_id=9, video_token_id=8)) == [9, 1, 8]


def test_prompt_without_placeholders_is_untouched():
    assert dedup([1, 2, 3], _Processor(image_token_id=9)) == [1, 2, 3]


def test_qwen_behaviour_is_unchanged():
    """Qwen was the only family this ever ran for; it must still get the same result."""
    prompt = [151652, 151655, 151655, 151655, 151653]

    assert dedup(prompt, _Processor(image_token_id=151655, class_name="Qwen2VLProcessor")) == [
        151652,
        151655,
        151653,
    ]


def test_processor_without_placeholder_ids_is_a_no_op():
    """A text-only processor declares none; touching the prompt would corrupt it."""
    prompt = [1, 1, 1]

    assert dedup(prompt, _Processor()) == prompt


def test_absent_processor_is_a_no_op():
    assert dedup([1, 1], None) == [1, 1]


def test_result_is_a_plain_list_of_ints():
    """The caller hands this straight to vLLM's TokensPrompt, which rejects arrays."""
    out = dedup(np.array([9, 9, 1]), _Processor(image_token_id=9))

    assert isinstance(out, list)
    assert all(isinstance(x, int) for x in out)


def test_legacy_name_still_resolves():
    from verl.workers.rollout.utils import qwen2_5_vl_dedup_image_tokens

    assert qwen2_5_vl_dedup_image_tokens is dedup
