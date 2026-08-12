# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from typing import Callable

_index_first_axis, _pad_input, _rearrange, _unpad_input = None, None, None, None


def _get_attention_functions() -> tuple[Callable, Callable, Callable, Callable]:
    """Dynamically import attention functions based on available hardware."""

    from verl.utils.device import is_torch_npu_available

    global _index_first_axis, _pad_input, _rearrange, _unpad_input

    if is_torch_npu_available(check_device=False):
        from verl.utils.npu_flash_attn_utils import index_first_axis, pad_input, rearrange, unpad_input
    else:
        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        except ImportError:
            # Same fallback the NPU branch already relies on, applied to CUDA. These are
            # padding helpers, not attention kernels -- transformers implements them in
            # plain torch and says so: "functionally equivalent to FA2's index_first_axis
            # and replaces the need to import it". Arities match, including the 5-tuple
            # unpad_input returns.
            #
            # Reachable now because the flash-attn package has no wheel for every torch:
            # upstream's newest release builds only up to torch 2.9, so on 2.11 the choice
            # is a source build or nothing. transformers itself handles the attention by
            # pulling kernels-community/flash-attn2 from the Hub, which leaves these four
            # helpers as the only thing still requiring the package. Without this, an
            # environment where attention works fine dies in the padding utilities.
            from einops import rearrange
            from transformers.modeling_flash_attention_utils import (
                _pad_input as pad_input,
                _unpad_input as unpad_input,
            )

            def index_first_axis(tensor, indices):
                # Deliberately not transformers' _index_first_axis, which is not the same
                # function despite the name: it reshapes away the leading *two* dimensions
                # because it is written for the (batch, seqlen, ...) tensor _unpad_input
                # hands it. Callers here pass a tensor already flattened to (total, ...),
                # which that version flattens a second time and returns one dimension
                # short -- tests/models/test_transformer.py catches it as an IndexError on
                # the following .transpose(0, 1), but a caller that does not transpose
                # would just get quietly wrong values.
                #
                # flash-attn's own forward is a gather equivalent to this indexing. It
                # wraps it in an autograd Function only to scatter the backward into a
                # zeros tensor instead of accumulating, which matches plain indexing for
                # the unique indices unpad_input produces.
                return tensor[indices]

    _index_first_axis, _pad_input, _rearrange, _unpad_input = index_first_axis, pad_input, rearrange, unpad_input

    return _index_first_axis, _pad_input, _rearrange, _unpad_input


def index_first_axis(*args, **kwargs):
    """
    Unified entry point for `index_first_axis` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.index_first_axis`
      - On NPU: `transformers.integrations.npu_flash_attention.index_first_axis`
        (falls back to `transformers.modeling_flash_attention_utils._index_first_axis`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def pad_input(*args, **kwargs):
    """
    Unified entry point for `pad_input` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.pad_input`
      - On NPU: `transformers.integrations.npu_flash_attention.pad_input`
        (falls back to `transformers.modeling_flash_attention_utils._pad_input`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    _, func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def rearrange(*args, **kwargs):
    """
    Unified entry point for `rearrange` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.rearrange`
      - On NPU: `transformers.integrations.npu_flash_attention.rearrange`
        (falls back to `einops.rearrange` if no dedicated NPU implementation exists).

    Users can call this function directly without worrying about the underlying device.
    """
    *_, func, _ = _get_attention_functions()
    return func(*args, **kwargs)


def unpad_input(*args, **kwargs):
    """
    Unified entry point for `unpad_input` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.unpad_input`
      - On NPU: `transformers.integrations.npu_flash_attention.unpad_input`
        (falls back to `transformers.modeling_flash_attention_utils._unpad_input`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    *_, func = _get_attention_functions()
    return func(*args, **kwargs)


__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]
