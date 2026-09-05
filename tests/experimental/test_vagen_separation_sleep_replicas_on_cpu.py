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

"""``init_workers`` must leave the rollout replicas asleep, on every trainer.

``fit`` runs ``_load_checkpoint`` and then ``checkpoint_manager.update_weights``
before anything has generated. The naive sync path issues
``rollout.resume(tags=["weights"])`` unconditionally, and SGLang implements resume
as ``self.offload_tags.remove(tag)``. If ``init_workers`` did not sleep the
replicas, that set is empty and the scheduler dies with ``KeyError: 'weights'`` --
surfacing on every rank only as "Failed to complete async request to
resume_memory_occupation after 3 attempts", which names neither the tag nor the
trainer.

``RayPPOTrainer.init_workers`` ends with the sleep;
``SeparateRayPPOTrainer.init_workers`` overrode it and dropped the call. Asserted
against the source rather than a live trainer, because reaching ``init_workers``
otherwise needs a Ray cluster, GPUs and a model download -- while the defect is
simply a missing line in an override.
"""

import inspect

from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

TRAINERS = [RayPPOTrainer, SeparateRayPPOTrainer]


def _init_workers_source(trainer_class):
    return inspect.getsource(trainer_class.init_workers)


def test_every_init_workers_sleeps_the_replicas():
    missing = [
        trainer_class.__name__
        for trainer_class in TRAINERS
        if "sleep_replicas()" not in _init_workers_source(trainer_class)
    ]
    assert not missing, (
        f"{missing} leave the rollout replicas awake; the first update_weights in fit() "
        f"then resumes a tag that was never released"
    )


def test_save_checkpoint_sleeps_the_replicas_around_the_save():
    """The engine must not hold its reservation while the save gathers parameters.

    _fit_update_weights resumes it earlier in the step and _fit_validate generates
    with it, so nothing has slept it by then. On a colocated node it is tens of GiB
    of the card, and the save OOMs non-deterministically -- fine at one step, dead
    at the next as the per-rank CUDA contexts grow.
    """
    source = inspect.getsource(SeparateRayPPOTrainer._fit_save_checkpoint)
    for call in ("sleep_replicas()", "_save_checkpoint()", "update_weights("):
        assert call in source, f"{call} missing from _fit_save_checkpoint"
    assert source.index("sleep_replicas()") < source.index("self._save_checkpoint()"), (
        "the replicas must be asleep before the save, not after"
    )
    assert source.index("self._save_checkpoint()") < source.index("update_weights("), (
        "the engine must be woken after the save; the next step opens with generation"
    )


def test_the_sleep_is_the_last_thing_init_workers_does():
    """It has to follow the CheckpointEngineManager it is called on."""
    for trainer_class in TRAINERS:
        source = _init_workers_source(trainer_class)
        assert source.index("CheckpointEngineManager(") < source.index("sleep_replicas()"), (
            f"{trainer_class.__name__} sleeps the replicas before building the manager"
        )
