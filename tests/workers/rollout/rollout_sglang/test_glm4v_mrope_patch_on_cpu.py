import torch

from verl.utils.sglang.glm4v_mrope_patch import _aligned_attention_mask


def test_all_one_mask_is_resized_to_preserved_prompt():
    input_ids = torch.tensor([[1, 2, 3, 4]])
    stale_mask = torch.ones((1, 3), dtype=torch.long)

    aligned = _aligned_attention_mask(input_ids, stale_mask)

    assert aligned.tolist() == [[1, 1, 1, 1]]


def test_matching_or_padded_masks_are_not_changed():
    input_ids = torch.tensor([[1, 2, 3, 4]])
    matching = torch.ones_like(input_ids)
    padded = torch.tensor([[1, 1, 0]])

    assert _aligned_attention_mask(input_ids, matching) is matching
    assert _aligned_attention_mask(input_ids, padded) is padded
