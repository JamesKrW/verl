# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""``SeparateRayPPOTrainer._create_critic_class`` must read the post-refactor names.

The engine refactor moved the engine config out from under ``model``: on the critic
dataclass ``model`` is an ``HFModelConfig`` with no ``fsdp_config``, the engine lives at
``engine``, and ``model_config`` does not exist. Reading the pre-refactor names raises
AttributeError for any run that enables a critic, and since nothing overrides this
method the whole separated lineage could only run critic-free algorithms.

Checked against the dataclass rather than a composed config: the mismatch is one of
attribute names, and introspecting the type keeps the test off both a Ray cluster and a
model download.
"""

import dataclasses
import inspect
import re

from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.workers.config.critic import FSDPCriticConfig

FIELDS = {f.name: f for f in dataclasses.fields(FSDPCriticConfig)}


def test_engine_config_is_not_under_model():
    """★ The regression: the trainer used to read `critic.model.fsdp_config`."""
    assert "engine" in FIELDS
    model_type = FIELDS["model"].type
    assert not any(f.name == "fsdp_config" for f in dataclasses.fields(model_type)), (
        f"{model_type.__name__} grew an fsdp_config again; re-check which name to read"
    )


def test_model_config_field_does_not_exist():
    assert "model" in FIELDS
    assert "model_config" not in FIELDS


def test_trainer_reads_only_fields_that_exist():
    """★ Catches the mismatch without a cluster: every ``orig_critic_cfg.<attr>`` the
    method reads has to be a real field."""
    src = inspect.getsource(SeparateRayPPOTrainer._create_critic_class)
    read = set(re.findall(r"orig_critic_cfg\.(\w+)", src))
    missing = sorted(a for a in read if a not in FIELDS)

    assert not missing, f"_create_critic_class reads {missing}, absent from FSDPCriticConfig"


def test_separated_lineage_does_not_override_critic_creation():
    """Pins why a bug here matters: it reaches every async trainer, because none of
    them supply their own critic creation."""
    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
    from verl.experimental.one_step_off_policy.ray_trainer import OneStepOffRayTrainer

    for cls in (OneStepOffRayTrainer, FullyAsyncTrainer):
        assert "_create_critic_class" not in vars(cls)
