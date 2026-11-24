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

import gc

import pytest
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils import gc_utils


@pytest.fixture(autouse=True)
def _restore_gc(monkeypatch):
    monkeypatch.setattr(gc_utils, "_frozen", False)
    yield
    gc.unfreeze()


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv(gc_utils.GC_FREEZE_ENV, raising=False)
    before = gc.get_freeze_count()

    assert not gc_utils.freeze_gc_after_first_step("step")
    assert gc.get_freeze_count() == before


def test_freezes_once_per_process(monkeypatch):
    monkeypatch.setenv(gc_utils.GC_FREEZE_ENV, "1")

    before = gc.get_freeze_count()
    assert gc_utils.freeze_gc_after_first_step("step")
    frozen = gc.get_freeze_count()
    assert frozen > before

    assert not gc_utils.freeze_gc_after_first_step("step")
    assert gc.get_freeze_count() == frozen


def test_flag_is_forwarded_to_workers(monkeypatch):
    monkeypatch.setenv(gc_utils.GC_FREEZE_ENV, "1")

    env_vars = get_ppo_ray_runtime_env(OmegaConf.create({"global_profiler": {"tool": None}}))["env_vars"]

    assert env_vars[gc_utils.GC_FREEZE_ENV] == "1"
