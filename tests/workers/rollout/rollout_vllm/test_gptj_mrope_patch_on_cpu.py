import inspect

import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

from verl.utils.vllm.gptj_mrope_patch import apply_gptj_mrope_compat_patch


def test_legacy_gptj_multimodal_rope_uses_correct_native_path():
    from vllm.model_executor.layers.rotary_embedding import mrope

    if "is_neox_style" in inspect.signature(mrope.triton_mrope).parameters:
        assert apply_gptj_mrope_compat_patch() is False
        return

    assert apply_gptj_mrope_compat_patch() is True
    assert apply_gptj_mrope_compat_patch() is True

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        rotary = mrope.MRotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=128,
            base=10_000,
            is_neox_style=False,
            dtype=torch.float32,
            mrope_section=[8, 12, 12],
        )
    positions = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 4]])
    query = torch.randn(3, 2 * 64)
    key = torch.randn(3, 64)

    expected_q, expected_k = rotary.forward_native(
        positions, query.clone(), key.clone()
    )
    actual_q, actual_k = rotary.forward_cuda(
        positions, query.clone(), key.clone()
    )

    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)
