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

"""Compatibility for GLM4V MRoPE with pre-tokenized SGLang requests.

SGLang's multimodal no-retokenize path replaces the processor's input ids with the
caller's exact ids after image expansion. A decode/re-tokenize boundary can differ by a
token, but SGLang 0.5.13 leaves the processor's old attention mask in place. GLM4V then
indexes the preserved ids with that stale mask and raises ``IndexError``.

TokenizerManager processes one unpadded request at a time, so an all-one mask carries
no information beyond its length. Resize only that safe case and leave every other
shape to SGLang's normal validation.
"""

from __future__ import annotations

import warnings

import torch


_PATCH_MARKER = "_verl_glm4v_mask_compat"


def _aligned_attention_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
    if attention_mask is None or tuple(attention_mask.shape) == tuple(input_ids.shape):
        return attention_mask
    if (
        input_ids.ndim != 2
        or attention_mask.ndim != 2
        or input_ids.shape[0] != 1
        or attention_mask.shape[0] != 1
        or not bool(torch.all(attention_mask == 1))
    ):
        return attention_mask
    return torch.ones(input_ids.shape, dtype=attention_mask.dtype, device=input_ids.device)


def apply_glm4v_mrope_mask_compat_patch() -> bool:
    """Make GLM4V's all-one attention mask match preserved prompt ids."""
    from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

    current = MRotaryEmbedding.get_rope_index_glm4v
    if getattr(current, _PATCH_MARKER, False):
        return True

    original = current
    warned = False

    def get_rope_index_glm4v(
        input_ids,
        hf_config,
        image_grid_thw,
        video_grid_thw,
        attention_mask,
        **kwargs,
    ):
        nonlocal warned
        aligned = _aligned_attention_mask(input_ids, attention_mask)
        if aligned is not attention_mask and not warned:
            warnings.warn(
                "Adjusted GLM4V attention_mask to match SGLang's preserved prompt ids.",
                stacklevel=2,
            )
            warned = True
        return original(
            input_ids=input_ids,
            hf_config=hf_config,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=aligned,
            **kwargs,
        )

    setattr(get_rope_index_glm4v, _PATCH_MARKER, True)
    setattr(get_rope_index_glm4v, "_verl_original", original)
    MRotaryEmbedding.get_rope_index_glm4v = staticmethod(get_rope_index_glm4v)
    return True


__all__ = ["apply_glm4v_mrope_mask_compat_patch"]
