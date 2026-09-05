import asyncio
from types import SimpleNamespace

import pytest
import sglang.srt.managers.tokenizer_manager as tokenizer_manager
import sglang.srt.utils.hf_transformers.processor as sglang_processor
import transformers
from sglang.srt.multimodal.processors.transformers_auto import (
    TransformersAutoMultimodalProcessor,
)

import verl.utils.sglang.internvl_processor_patch as internvl_patch
from verl.utils.sglang.internvl_processor_patch import (
    _PROCESSOR_MARKER,
    _auto_processor_kwargs,
    _restore_text_prompt,
    apply_internvl35_processor_compat_patch,
)
from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer


class Tokenizer:
    def decode(self, input_ids, *, skip_special_tokens):
        assert input_ids == [1, 2, 3]
        assert not skip_special_tokens
        return "before<img><IMG_CONTEXT></img>after"

    def convert_tokens_to_ids(self, token):
        return {"<img>": 11, "<IMG_CONTEXT>": 12, "</img>": 13}[token]


class InternVLProcessor:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.start_image_token = "<img>"
        self.image_token = "<IMG_CONTEXT>"
        self.end_image_token = "</img>"
        self.start_image_token_id = -1
        self.image_token_id = -1
        self.end_image_token_id = -1


def _isolate_patch(monkeypatch, get_processor):
    async def process_mm_data_async(self, image_data, audio_data, input_text, request_obj, **kwargs):
        return input_text

    monkeypatch.setattr(tokenizer_manager, "get_processor", get_processor)
    monkeypatch.setattr(
        TransformersAutoMultimodalProcessor,
        "process_mm_data_async",
        process_mm_data_async,
    )


def test_auto_processor_kwargs_remove_sglang_only_options():
    assert _auto_processor_kwargs(
        {
            "tokenizer_mode": "auto",
            "tokenizer_backend": "huggingface",
            "tokenizer_revision": "main",
            "trust_remote_code": False,
            "use_fast": True,
        }
    ) == {
        "revision": "main",
        "trust_remote_code": False,
        "use_fast": True,
    }


def test_patch_rebuilds_full_processor_and_refreshes_token_ids(monkeypatch):
    tokenizer = Tokenizer()
    full_processor = InternVLProcessor()
    calls = {}

    def original(tokenizer_name, *args, **kwargs):
        calls["original"] = (tokenizer_name, args, kwargs)
        return tokenizer

    def from_pretrained(tokenizer_name, *args, **kwargs):
        calls["auto"] = (tokenizer_name, args, kwargs)
        return full_processor

    _isolate_patch(monkeypatch, original)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", from_pretrained)
    monkeypatch.setattr(
        sglang_processor,
        "resolve_runai_obj_uri",
        lambda name: "/resolved/model" if name.startswith("runai://") else name,
    )

    assert apply_internvl35_processor_compat_patch()
    result = tokenizer_manager.get_processor(
        "runai://OpenGVLab/InternVL3_5-2B-hf",
        tokenizer_mode="auto",
        tokenizer_backend="huggingface",
        revision="main",
        use_fast=True,
    )

    assert result is full_processor
    assert result.tokenizer is tokenizer
    assert result.start_image_token_id == 11
    assert result.image_token_id == 12
    assert result.end_image_token_id == 13
    assert getattr(result, _PROCESSOR_MARKER)
    assert calls["auto"] == ("/resolved/model", (), {"revision": "main", "use_fast": True})


def test_patch_marks_an_existing_full_processor_without_reloading(monkeypatch):
    full_processor = InternVLProcessor(Tokenizer())
    _isolate_patch(monkeypatch, lambda *args, **kwargs: full_processor)

    def unexpected_reload(*args, **kwargs):
        raise AssertionError("AutoProcessor should not reload a full processor")

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", unexpected_reload)
    apply_internvl35_processor_compat_patch()

    assert tokenizer_manager.get_processor("OpenGVLab/InternVL3-1B-hf") is full_processor
    assert getattr(full_processor, _PROCESSOR_MARKER)


def test_patch_is_idempotent(monkeypatch):
    _isolate_patch(monkeypatch, lambda *args, **kwargs: InternVLProcessor(Tokenizer()))
    apply_internvl35_processor_compat_patch()
    first_get_processor = tokenizer_manager.get_processor
    first_process = TransformersAutoMultimodalProcessor.process_mm_data_async

    apply_internvl35_processor_compat_patch()

    assert tokenizer_manager.get_processor is first_get_processor
    assert TransformersAutoMultimodalProcessor.process_mm_data_async is first_process


def test_async_patch_only_rewrites_marked_internvl_processors(monkeypatch):
    _isolate_patch(monkeypatch, lambda *args, **kwargs: InternVLProcessor(Tokenizer()))
    apply_internvl35_processor_compat_patch()

    marked = InternVLProcessor(Tokenizer())
    setattr(marked, _PROCESSOR_MARKER, True)
    unmarked = InternVLProcessor(Tokenizer())
    method = TransformersAutoMultimodalProcessor.process_mm_data_async

    rewritten = asyncio.run(method(SimpleNamespace(_processor=marked), None, None, [1, 2, 3], object()))
    unchanged = asyncio.run(method(SimpleNamespace(_processor=unmarked), None, None, [1, 2, 3], object()))

    assert rewritten == "before<IMG_CONTEXT>after"
    assert unchanged == [1, 2, 3]


def test_restore_text_prompt_is_defensive():
    processor = SimpleNamespace(tokenizer=Tokenizer())
    assert _restore_text_prompt([1, 2, 3], processor) == "before<img><IMG_CONTEXT></img>after"
    assert _restore_text_prompt("already text", processor) == "already text"


def test_server_activates_patch_for_internvl(monkeypatch):
    called = []
    monkeypatch.setattr(
        internvl_patch,
        "apply_internvl35_processor_compat_patch",
        lambda: called.append(True),
    )
    server = object.__new__(SGLangHttpServer)
    server._disaggregation_role = "null"
    server.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="internvl"))
    server.nnodes = 2
    server.node_rank = 1

    with pytest.raises(AssertionError, match="non-master node"):
        asyncio.run(server.launch_server())

    assert called == [True]
