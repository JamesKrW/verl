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
"""GLM-4V's Ulysses attention patch follows the transformers 5.x RoPE API."""

from unittest import mock

import torch
from transformers.models.glm4v.modeling_glm4v import apply_rotary_pos_emb

from verl.models.transformers.glm4v import glm4v_attn_forward


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.num_key_value_heads = 1
        self.num_key_value_groups = 2
        self.head_dim = 4
        self.hidden_size = 8
        self.attention_dropout = 0.0
        self.is_causal = True
        self.training = False
        self.q_proj = torch.nn.Linear(8, 8, bias=False)
        self.k_proj = torch.nn.Linear(8, 4, bias=False)
        self.v_proj = torch.nn.Linear(8, 4, bias=False)
        self.o_proj = torch.nn.Identity()


def test_attention_uses_transformers5_precomputed_rope():
    torch.manual_seed(0)
    attention = _Attention()
    hidden = torch.randn(1, 3, 8)
    cos = torch.randn(1, 3, 4)
    sin = torch.randn(1, 3, 4)
    captured = {}

    def fake_flash(query, key, value, *args, **kwargs):
        captured.update(query=query, key=key, value=value)
        return query

    with mock.patch("verl.models.transformers.glm4v._custom_flash_attention_forward", new=fake_flash):
        output, _ = glm4v_attn_forward(
            attention,
            hidden,
            position_ids=torch.arange(3).unsqueeze(0),
            position_embeddings=(cos, sin),
        )

    query = attention.q_proj(hidden).view(1, 3, 2, 4).transpose(1, 2)
    key = attention.k_proj(hidden).view(1, 3, 1, 4).transpose(1, 2)
    expected_query, expected_key = apply_rotary_pos_emb(query, key, cos, sin)
    expected_key = expected_key.repeat_interleave(2, dim=1)

    torch.testing.assert_close(captured["query"], expected_query.transpose(1, 2))
    torch.testing.assert_close(captured["key"], expected_key.transpose(1, 2))
    assert output.shape == (1, 3, 8)
