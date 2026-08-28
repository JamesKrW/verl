import pickle

import tensordict
import torch
from packaging.version import parse as parse_version
from tensordict import TensorDict

from verl.models.transformers.qwen2_vl import normalize_position_ids_layout, process_position_ids
from verl.utils.tensordict_utils import chunk_tensordict, maybe_fix_3d_position_ids
from verl.utils.transformers_compat import is_transformers_version_in_range
from verl.workers.utils.padding import left_right_2_no_padding


def test_packed_qwen_position_ids_are_normalized_to_rope_first():
    batch_first = torch.zeros((1, 7, 4), dtype=torch.long)


    normalized = normalize_position_ids_layout(batch_first, sequence_length=7)

    assert normalized.shape == (4, 1, 7)


def test_packed_qwen_position_ids_keep_rope_first_layout():
    rope_first = torch.zeros((4, 1, 7), dtype=torch.long)

    assert normalize_position_ids_layout(rope_first, sequence_length=7) is rope_first


def test_transformers_5_keeps_the_text_channel_for_packed_attention():
    if not is_transformers_version_in_range(min_version="5.0.0"):
        return

    packed_position_ids = torch.zeros((4, 1, 7), dtype=torch.long)

    assert process_position_ids(packed_position_ids).shape == (4, 1, 7)


def test_new_tensordict_does_not_rewrite_the_jagged_position_axis():
    if parse_version(tensordict.__version__) <= parse_version("0.10.0"):
        return

    width, valid = 20, 11
    data = TensorDict(
        {
            "input_ids": torch.zeros((1, width), dtype=torch.long),
            "attention_mask": torch.cat(
                [torch.zeros((1, width - valid), dtype=torch.long), torch.ones((1, valid), dtype=torch.long)],
                dim=1,
            ),
            "response_mask": torch.zeros((1, 10), dtype=torch.long),
            "position_ids": torch.arange(4 * width).reshape(1, 4, width),
        },
        batch_size=1,
    )
    data = pickle.loads(pickle.dumps(left_right_2_no_padding(data)))

    maybe_fix_3d_position_ids(data)
    position_ids = chunk_tensordict(data, 1)[0]["position_ids"].values()

    assert position_ids.shape == (4, valid)
