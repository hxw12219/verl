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
"""Keep a process's warmed-up heap out of Python's full garbage collections.

The trainer driver, training workers and agent-loop workers each hold millions of long-lived objects after the first
step (dataset rows, modules, tokenizers, kernel caches). Every full (generation-2) collection rescans all of them, which
can stall a step for a second or more. Freezing once after the first step moves every live object to the permanent
generation, so later collections only scan objects allocated since. vLLM does the same after its warm-up
(``vllm.utils.gc_utils.freeze_gc_heap``).

Frozen objects are still freed as soon as their reference count drops to zero. Only reference cycles that are alive at
the freeze are never collected, so the cost is bounded by the first step's live cycles.
"""

import gc
import logging
import os

GC_FREEZE_ENV = "VERL_GC_FREEZE_AFTER_FIRST_STEP"

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_frozen = False


def freeze_gc_after_first_step(stage: str) -> bool:
    """Collect garbage and freeze the survivors on the first call in this process.

    Does nothing unless ``VERL_GC_FREEZE_AFTER_FIRST_STEP=1`` (set from ``trainer.gc_freeze_after_first_step``), and
    nothing after the first call. Returns whether this call froze the heap.
    """
    global _frozen
    if _frozen or os.environ.get(GC_FREEZE_ENV, "0") != "1":
        return False
    _frozen = True
    gc.collect()
    gc.freeze()
    logger.info("gc: froze %d objects after the first %s (pid %d)", gc.get_freeze_count(), stage, os.getpid())
    return True
