# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""An unknown VLM must not be routed through the text-only fused forward."""

from types import SimpleNamespace

import pytest
import torch

from verl.models.transformers.monkey_patch import patch_forward_with_backends


class _UnknownVLM(torch.nn.Module):
    config = SimpleNamespace(model_type="internvl", vision_config=SimpleNamespace())

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        return input_ids, pixel_values


def test_unknown_vlm_refuses_the_text_only_fused_forward():
    model = _UnknownVLM()
    native_forward = type(model).forward

    with pytest.raises(NotImplementedError, match="use_fused_kernels=False"):
        patch_forward_with_backends(model, use_fused_kernels=True, fused_kernels_backend="torch")

    assert type(model).forward is native_forward
    input_ids = torch.tensor([[1, 2]])
    pixel_values = torch.randn(1, 3, 2, 2)
    returned_ids, returned_pixels = model(input_ids=input_ids, pixel_values=pixel_values)
    assert returned_ids is input_ids
    assert returned_pixels is pixel_values
