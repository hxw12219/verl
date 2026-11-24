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

import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.workers import engine_workers
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer


class _FakeDevice:
    def __init__(self):
        self.empty_cache_calls = 0

    def empty_cache(self):
        self.empty_cache_calls += 1

    def synchronize(self):
        pass

    def ipc_collect(self):
        pass


class _ResidentRollout:
    def __init__(self):
        self.updates = 0

    async def resume(self, tags):
        raise AssertionError("a resident rollout is never resumed")

    async def update_weights(self, weights, **kwargs):
        self.updates += 1


@pytest.mark.parametrize("keep_trainer_cache, expected_releases", [(False, 2), (True, 0)])
def test_naive_sync_to_resident_rollout_keeps_trainer_cache(monkeypatch, keep_trainer_cache, expected_releases):
    device = _FakeDevice()
    monkeypatch.setattr(engine_workers, "get_torch_device", lambda: device)
    monkeypatch.setattr(engine_workers, "set_expandable_segments", lambda enable: None)
    monkeypatch.setattr(engine_workers, "log_gpu_memory_usage", lambda *args, **kwargs: None)
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.config = OmegaConf.create(
        {
            "rollout": {
                "name": "vllm",
                "free_cache_engine": False,
                "checkpoint_engine": {"backend": "naive", "keep_trainer_cache": keep_trainer_cache},
            }
        }
    )
    worker.actor = SimpleNamespace(
        engine=SimpleNamespace(
            is_param_offload_enabled=False,
            get_per_tensor_param=lambda **kwargs: (iter([]), None),
        )
    )
    worker.rollout = _ResidentRollout()
    worker.layered_summon = False
    worker.peft_merge = False
    worker.lora_resync_base = False
    worker.base_sync_done = True

    asyncio.run(ActorRolloutRefWorker.update_weights.__wrapped__(worker, global_steps=1, mode="auto"))

    assert worker.rollout.updates == 1
    assert device.empty_cache_calls == expected_releases


@pytest.mark.parametrize("release_cache, expected_releases", [(True, 1), (False, 0)])
def test_bucketed_sender_cleanup_release_cache(monkeypatch, release_cache, expected_releases):
    device = _FakeDevice()
    monkeypatch.setattr(bucketed_weight_transfer, "get_torch_device", lambda: device)
    sender = bucketed_weight_transfer.BucketedWeightSender(
        zmq_handle="tcp://127.0.0.1:0", bucket_size_mb=1, release_cache=release_cache
    )

    sender._cleanup()

    assert device.empty_cache_calls == expected_releases
