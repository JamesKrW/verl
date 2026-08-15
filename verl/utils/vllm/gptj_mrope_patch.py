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

"""Compatibility for GPT-J-style multimodal RoPE in older vLLM releases.

vLLM's Triton MRoPE kernel historically hard-coded NeoX-style pairs.  GLM-V uses
GPT-J/interleaved pairs, so image prompts receive incorrect rotations while text-only
prompts remain unaffected.  Upstream fixes this in vllm-project/vllm#42765.

Until that API is present, use vLLM's own correct PyTorch implementation only for the
affected 2-D GPT-J path.  Qwen and every NeoX-style model keep the original Triton path.
"""

from __future__ import annotations

import inspect


_PATCH_MARKER = "_verl_gptj_mrope_compat"


def apply_gptj_mrope_compat_patch() -> bool:
    """Apply the compatibility patch when the installed vLLM still needs it.

    Returns ``True`` when the old implementation was detected (including an already
    patched process), and ``False`` when upstream already accepts ``is_neox_style``.
    """
    from vllm.model_executor.layers.rotary_embedding import mrope

    if "is_neox_style" in inspect.signature(mrope.triton_mrope).parameters:
        return False

    cls = mrope.MRotaryEmbedding
    current = cls.forward_cuda
    if getattr(current, _PATCH_MARKER, False):
        return True

    original = current

    def forward_cuda(
        self,
        positions,
        query,
        key=None,
        offsets=None,
    ):
        if positions.ndim == 2 and not self.is_neox_style:
            return self.forward_native(positions, query, key, offsets)
        return original(self, positions, query, key, offsets)

    setattr(forward_cuda, _PATCH_MARKER, True)
    setattr(forward_cuda, "_verl_original", original)
    cls.forward_cuda = forward_cuda
    return True


__all__: list[str] = ["apply_gptj_mrope_compat_patch"]
