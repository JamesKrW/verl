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

"""Compatibility for native Transformers InternVL 3.5 in SGLang 0.5.13.

SGLang special-cases paths containing ``InternVL3_5`` and loads only an
``AutoTokenizer``.  That is correct for its legacy ``InternVLChatModel`` but
not for Transformers' native ``InternVLForConditionalGeneration`` backend,
whose generic multimodal adapter requires an ``InternVLProcessor``.
"""

from __future__ import annotations

import logging


_PATCH_MARKER = "_verl_internvl35_processor_compat"
_PROCESSOR_MARKER = "_verl_native_internvl_processor"
_IMAGE_TOKEN_ATTRS = (
    ("image_token", "image_token_id"),
    ("start_image_token", "start_image_token_id"),
    ("end_image_token", "end_image_token_id"),
)
logger = logging.getLogger(__name__)


def _auto_processor_kwargs(kwargs: dict) -> dict:
    """Drop SGLang-only tokenizer options before calling AutoProcessor."""
    processor_kwargs = dict(kwargs)
    processor_kwargs.pop("tokenizer_mode", None)
    processor_kwargs.pop("tokenizer_backend", None)
    tokenizer_revision = processor_kwargs.pop("tokenizer_revision", None)
    if tokenizer_revision is not None:
        processor_kwargs.setdefault("revision", tokenizer_revision)
    return processor_kwargs


def _restore_text_prompt(input_text, processor):
    """Decode pre-tokenized input and undo VAGEN's one-token image compression."""
    if not isinstance(input_text, list) or not all(isinstance(token, int) for token in input_text):
        return input_text

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "decode"):
        return input_text
    text = tokenizer.decode(input_text, skip_special_tokens=False)
    image_token = getattr(processor, "image_token", None)
    start_token = getattr(processor, "start_image_token", None)
    end_token = getattr(processor, "end_image_token", None)
    if not all(isinstance(token, str) and token for token in (image_token, start_token, end_token)):
        return text
    wrapped_image_token = f"{start_token}{image_token}{end_token}"
    return text.replace(wrapped_image_token, image_token)


def _is_internvl_processor(processor) -> bool:
    return hasattr(processor, "tokenizer") and all(
        isinstance(getattr(processor, token_attr, None), str)
        for token_attr, _ in _IMAGE_TOKEN_ATTRS
    )


def _replace_tokenizer(processor, tokenizer) -> None:
    """Install SGLang's tokenizer and refresh processor-cached image token ids."""
    processor.tokenizer = tokenizer
    for token_attr, id_attr in _IMAGE_TOKEN_ATTRS:
        token = getattr(processor, token_attr, None)
        if not isinstance(token, str) or not token:
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            setattr(processor, id_attr, token_id)


def apply_internvl35_processor_compat_patch() -> bool:
    """Return a full HF processor for native InternVL 3.5 SGLang servers."""
    from transformers import AutoProcessor

    import sglang.srt.managers.tokenizer_manager as tokenizer_manager

    try:
        from sglang.srt.multimodal.processors.transformers_auto import (
            TransformersAutoMultimodalProcessor,
        )
        from sglang.srt.utils.hf_transformers.processor import resolve_runai_obj_uri
    except ImportError as exc:
        raise RuntimeError(
            "native InternVL requires SGLang >=0.5.13 with its Transformers multimodal backend"
        ) from exc

    current = tokenizer_manager.get_processor
    if not getattr(current, _PATCH_MARKER, False):
        original = current
        warned = False

        def get_processor(tokenizer_name, *args, **kwargs):
            nonlocal warned
            processor = original(tokenizer_name, *args, **kwargs)
            if _is_internvl_processor(processor):
                setattr(processor, _PROCESSOR_MARKER, True)
                return processor
            if hasattr(processor, "tokenizer"):
                return processor

            full_processor = AutoProcessor.from_pretrained(
                resolve_runai_obj_uri(tokenizer_name),
                *args,
                **_auto_processor_kwargs(kwargs),
            )
            if not _is_internvl_processor(full_processor):
                raise RuntimeError(
                    f"AutoProcessor returned {type(full_processor).__name__}, expected an InternVL processor"
                )
            # Keep SGLang's tokenizer fixes and additional stop-token metadata.
            _replace_tokenizer(full_processor, processor)
            setattr(full_processor, _PROCESSOR_MARKER, True)
            if not warned:
                logger.warning("Loaded the full InternVL processor for SGLang.")
                warned = True
            return full_processor

        setattr(get_processor, _PATCH_MARKER, True)
        setattr(get_processor, "_verl_original", original)
        tokenizer_manager.get_processor = get_processor

    current_process = TransformersAutoMultimodalProcessor.process_mm_data_async
    if not getattr(current_process, _PATCH_MARKER, False):
        original_process = current_process

        async def process_mm_data_async(
            self, image_data, audio_data, input_text, request_obj, **kwargs
        ):
            processor = getattr(self, "_processor", None)
            if getattr(processor, _PROCESSOR_MARKER, False):
                input_text = _restore_text_prompt(input_text, processor)
            return await original_process(
                self, image_data, audio_data, input_text, request_obj, **kwargs
            )

        setattr(process_mm_data_async, _PATCH_MARKER, True)
        setattr(process_mm_data_async, "_verl_original", original_process)
        TransformersAutoMultimodalProcessor.process_mm_data_async = process_mm_data_async
    return True


__all__ = ["apply_internvl35_processor_compat_patch"]
