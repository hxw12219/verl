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

"""``response_only_lm_head`` with real FSDP and FSDP2 engines on 2 GPUs: same log-probs, entropy and gradients."""

import os
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, Qwen3Config

from verl import DataProto
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig, FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding, response_logit_rows

WORLD_SIZE = 2
VOCAB_SIZE = 1024
PROMPT_LENS, RESPONSE_LENS = [5, 9, 3, 7], [4, 2, 6, 3]


def _batch(use_fused_kernels):
    """Left-padded prompts and right-padded responses of different lengths, in the no-padding layout."""
    prompt_width, response_width = max(PROMPT_LENS), max(RESPONSE_LENS)
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(1, VOCAB_SIZE, (len(PROMPT_LENS), prompt_width + response_width), generator=generator)
    attention_mask = torch.zeros_like(input_ids)
    response_mask = torch.zeros(len(PROMPT_LENS), response_width, dtype=torch.long)
    for i, (prompt_len, response_len) in enumerate(zip(PROMPT_LENS, RESPONSE_LENS, strict=True)):
        attention_mask[i, prompt_width - prompt_len : prompt_width + response_len] = 1
        response_mask[i, :response_len] = 1
    data = DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :prompt_width],
            "responses": input_ids[:, prompt_width:],
            "attention_mask": attention_mask,
            "position_ids": (attention_mask.cumsum(-1) - 1).clamp(min=0),
            "response_mask": response_mask,
            "old_log_probs": torch.zeros(response_mask.shape),
            "advantages": torch.randn(response_mask.shape, generator=generator),
        },
        meta_info={"temperature": 0.7},
    )
    data = left_right_2_no_padding(data.to_tensordict())
    tu.assign_non_tensor(
        data,
        global_batch_size=len(PROMPT_LENS) * WORLD_SIZE,
        use_remove_padding=True,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=2,
        use_fused_kernels=use_fused_kernels,
        calculate_entropy=True,
        return_model_output=True,
    )
    return data


def _forward_backward(model_path, strategy, fused_backend, response_only_lm_head):
    from verl.workers.engine import EngineRegistry

    use_fused_kernels = fused_backend is not None
    model_config = HFModelConfig(
        path=model_path,
        load_tokenizer=False,
        override_config={"attn_implementation": "sdpa"},
        use_remove_padding=True,
        use_fused_kernels=use_fused_kernels,
        fused_kernel_options={"impl_backend": fused_backend} if fused_backend else {},
    )
    engine_config = FSDPEngineConfig(
        forward_only=False,
        fsdp_size=WORLD_SIZE,
        strategy=strategy,
        use_remove_padding=True,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=2,
        mixed_precision={"param_dtype": "fp32", "reduce_dtype": "fp32", "buffer_dtype": "fp32"},
        response_only_lm_head=response_only_lm_head,
    )
    engine = EngineRegistry.new(
        model_type="language_model",
        backend=strategy,
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=FSDPOptimizerConfig(lr=1e-3, lr_scheduler_type="constant"),
        checkpoint_config=CheckpointConfig(),
    )
    engine.initialize()
    loss_function = partial(
        ppo_loss, config=ActorConfig(strategy=strategy, rollout_n=1, ppo_micro_batch_size_per_gpu=-1)
    )
    with engine.train_mode(zero_grad_on_exit=False):
        engine.optimizer_zero_grad()
        output = engine.forward_backward_batch(_batch(use_fused_kernels), loss_function=loss_function)
    grads = []
    for parameter in engine.module.parameters():
        grad = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
        grads.append(grad.detach().float().clone())
    return {name: value.values().detach().float().cpu() for name, value in output["model_output"].items()}, grads


def _worker(rank, rendezvous_file, model_path, strategy, fused_backend):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method=f"file://{rendezvous_file}", rank=rank, world_size=WORLD_SIZE)
    full, full_grads = _forward_backward(model_path, strategy, fused_backend, response_only_lm_head=False)
    kept, kept_grads = _forward_backward(model_path, strategy, fused_backend, response_only_lm_head=True)

    reference = _batch(fused_backend is not None)
    assert set(kept) == set(full) and {"log_probs", "entropy"} <= set(full)
    for name in full:
        torch.testing.assert_close(
            no_padding_2_padding(kept[name], reference), no_padding_2_padding(full[name], reference)
        )
    # Only response rows are projected; the response-only output keeps zeros at every other row.
    other_rows = torch.ones(full["log_probs"].shape[0], dtype=torch.bool)
    other_rows[response_logit_rows(reference)] = False
    assert (kept["log_probs"][other_rows] == 0).all() and (full["log_probs"][other_rows] != 0).any()
    assert any(grad.abs().sum() > 0 for grad in full_grads)
    for kept_grad, full_grad in zip(kept_grads, full_grads, strict=True):
        torch.testing.assert_close(kept_grad, full_grad)

    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < WORLD_SIZE, reason="requires 2 GPUs")
@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
# Fused kernels read lm_head.weight directly, which FSDP2 leaves sharded unless the embeddings are tied.
@pytest.mark.parametrize(
    "fused_backend, tie_word_embeddings",
    [(None, False), ("torch", True), ("triton", True)],
    ids=["eager", "fused_torch", "fused_triton"],
)
def test_response_only_lm_head_matches_full_lm_head(strategy, fused_backend, tie_word_embeddings, tmp_path):
    config = Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        tie_word_embeddings=tie_word_embeddings,
    )
    model_path = os.path.join(tmp_path, "model")
    AutoModelForCausalLM.from_config(config).save_pretrained(model_path)
    mp.spawn(
        _worker,
        args=(str(tmp_path / "rendezvous"), model_path, strategy, fused_backend),
        nprocs=WORLD_SIZE,
        join=True,
    )
