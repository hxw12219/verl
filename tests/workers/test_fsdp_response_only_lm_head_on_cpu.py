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

"""``response_only_lm_head`` in the FSDP engine, on CPU with a tiny Qwen3."""

import copy
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from transformers import Qwen3Config, Qwen3ForCausalLM

import verl.utils.experimental.torch_functional as experimental_F
import verl.utils.torch_functional as verl_F
from verl.models.transformers.dense_common import _keep_rows, forward_with_torch_backend
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead, _scatter_rows
from verl.workers.utils.padding import no_padding_2_padding, response_logit_rows

PROMPTS = [[5, 6, 7], [8, 9, 10, 11, 12], [13, 14]]
RESPONSES = [[20, 21], [22], [23, 24, 25]]
# Packed lengths 5, 6, 5; each sequence is read at rows [end - R - 1, end - 1).
RESPONSE_ROWS = [2, 3, 9, 12, 13, 14]
PACKED_TOKENS = 16


def _jagged(rows):
    return torch.nested.as_nested_tensor([torch.tensor(row) for row in rows], layout=torch.jagged)


def _micro_batch(nested_prompts=True, temperature=0.7, **flags):
    sequences = [p + r for p, r in zip(PROMPTS, RESPONSES, strict=True)]
    fields = {"input_ids": _jagged(sequences), "position_ids": _jagged([list(range(len(s))) for s in sequences])}
    if nested_prompts:
        fields.update(prompts=_jagged(PROMPTS), responses=_jagged(RESPONSES))
    else:  # left-padded prompts, right-padded responses
        pw, rw = max(map(len, PROMPTS)), max(map(len, RESPONSES))
        fields["prompts"] = torch.tensor([[0] * (pw - len(p)) + p for p in PROMPTS])
        fields["responses"] = torch.tensor([r + [0] * (rw - len(r)) for r in RESPONSES])
        fields["attention_mask"] = torch.tensor(
            [
                [0] * (pw - len(p)) + [1] * len(p + r) + [0] * (rw - len(r))
                for p, r in zip(PROMPTS, RESPONSES, strict=True)
            ]
        )
    micro_batch = TensorDict(fields, batch_size=[])
    if isinstance(temperature, torch.Tensor):
        micro_batch["temperature"] = temperature
    else:
        flags["temperature"] = temperature
    tu.assign_non_tensor(micro_batch, **{"use_remove_padding": True, "pad_mode": DatasetPadMode.NO_PADDING, **flags})
    return micro_batch


def _engine(response_only_lm_head, model=None, **attributes):
    """Engine with only what prepare_model_inputs/outputs read; a real one needs a process group."""
    engine = object.__new__(FSDPEngineWithLMHead)
    config = SimpleNamespace(
        response_only_lm_head=response_only_lm_head,
        entropy_checkpointing=False,
        entropy_from_logits_with_chunking=False,
    )
    defaults = {"engine_config": config, "module": model, "pad_to_length": False, "pad_to_length_bucket": 1024}
    defaults.update(use_ulysses_sp=False, forward_accepts_logits_to_keep=True)
    engine.__dict__.update(defaults, compute_entropy_from_logits=verl_F.entropy_from_logits)
    engine.__dict__.update(attributes)
    return engine


def _tiny_model():
    config = Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, tie_word_embeddings=True,
    )  # fmt: skip
    return Qwen3ForCausalLM(config).double()


@pytest.mark.parametrize("nested_prompts", [True, False])
def test_rows_are_exactly_the_rows_no_padding_2_padding_reads(nested_prompts):
    micro_batch = _micro_batch(nested_prompts=nested_prompts)
    rows = response_logit_rows(micro_batch)
    assert rows.tolist() == RESPONSE_ROWS
    full = torch.randn(PACKED_TOKENS, 4)
    restored = torch.zeros_like(full).index_copy(0, rows, full[rows])
    assert torch.equal(no_padding_2_padding(restored, micro_batch), no_padding_2_padding(full, micro_batch))


@pytest.mark.parametrize("temperature", [1.0, torch.tensor([0.7, 1.3, 0.9])], ids=["unit", "per_sample"])
@pytest.mark.parametrize("calculate_entropy", [False, True])
@pytest.mark.parametrize("pad_to_length", [False, True])
@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused_torch"])
def test_response_only_lm_head_matches_the_full_projection(
    temperature, calculate_entropy, pad_to_length, fused, monkeypatch
):
    if fused and isinstance(temperature, torch.Tensor):
        pytest.skip("fused kernels take one scalar temperature")
    # The flash-attn cross-entropy is CUDA-only.
    monkeypatch.setenv("VERL_DISABLE_FLASH_ATTN_CE", "1")
    monkeypatch.setattr(experimental_F, "_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE", False)
    torch.manual_seed(0)
    base_model = _tiny_model()
    outputs, gradients = {}, {}
    for response_only in (False, True):
        model = copy.deepcopy(base_model)
        if fused:  # the forward apply_monkey_patch installs with use_fused_kernels
            model.forward = forward_with_torch_backend.__get__(model)
        micro_batch = _micro_batch(
            temperature=temperature,
            calculate_entropy=calculate_entropy,
            calculate_sum_pi_squared=not fused,
            use_fused_kernels=fused,
        )
        # A 12-token bucket pads the 16 packed tokens to 24.
        engine = _engine(response_only, model, pad_to_length=pad_to_length, pad_to_length_bucket=12)
        model_inputs, output_args = engine.prepare_model_inputs(micro_batch)
        output = model(**model_inputs, use_cache=False)
        scored = output.log_probs if fused else output.logits
        assert scored.shape[1] == (len(RESPONSE_ROWS) if response_only else model_inputs["input_ids"].shape[1])
        result = engine.prepare_model_outputs(output, output_args, micro_batch, logits_processor_func=None)
        outputs[response_only] = {name: no_padding_2_padding(value, micro_batch) for name, value in result.items()}
        sum(value.sum() for value in outputs[response_only].values()).backward()
        gradients[response_only] = [parameter.grad for parameter in model.parameters()]
    torch.testing.assert_close(outputs[True], outputs[False])
    torch.testing.assert_close(gradients[True], gradients[False])


@pytest.mark.parametrize(
    "logits_to_keep, rows", [(0, slice(None)), (2, slice(4, None)), (torch.tensor([1, 3, 4]), [1, 3, 4])]
)
def test_fused_forward_keeps_rows_like_transformers(logits_to_keep, rows):
    hidden, labels = torch.arange(24.0).view(1, 6, 4), torch.arange(6).view(1, 6)
    kept_hidden, kept_labels = _keep_rows(hidden, labels, logits_to_keep)
    assert torch.equal(kept_hidden, hidden[:, rows]) and torch.equal(kept_labels, labels[:, rows])


def test_flag_off_keeps_the_full_projection():
    for engine in (_engine(False), _engine(None, engine_config=SimpleNamespace())):
        model_inputs, output_args = engine.prepare_model_inputs(_micro_batch())
        assert "logits_to_keep" not in model_inputs and "response_logit_rows" not in output_args


@pytest.mark.parametrize(
    "flags, attributes",
    [
        ({"use_remove_padding": False}, {}),
        ({}, {"use_ulysses_sp": True}),
        ({"distillation_use_topk": True}, {}),
        ({"score_centering": True}, {}),
        ({}, {"forward_accepts_logits_to_keep": False}),
    ],
    ids=["padded", "sequence_parallel", "distillation", "score_centering", "forward_without_logits_to_keep"],
)
def test_unsupported_paths_are_refused(flags, attributes):
    with pytest.raises(ValueError, match="response_only_lm_head is not supported"):
        _engine(True, **attributes).prepare_model_inputs(_micro_batch(**flags))


def test_a_forward_that_ignores_logits_to_keep_is_reported():
    with pytest.raises(RuntimeError, match="logits_to_keep"):
        _scatter_rows(torch.zeros(PACKED_TOKENS), torch.tensor(RESPONSE_ROWS), PACKED_TOKENS)
