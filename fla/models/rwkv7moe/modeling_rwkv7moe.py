# coding=utf-8
# Copyright 2024 The RWKV team and HuggingFace Inc. team.
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
"""PyTorch rwkv7Moe World model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_ninja_available,
    is_torch_cuda_available,
    logging,
)

from fla.models.rwkv7moe.configuration_rwkv7moe import rwkv7MoeConfig
try:
    from fla.ops.rwkv7 import fused_recurrent_rwkv7
except ImportError:
    print("Required module is not installed. Please install it using the following commands:")
    print("pip install -U git+https://github.com/sustcsonglin/flash-linear-attention")
    print("Additionally, ensure you have the correct version of Triton installed:")
    print("pip install triton==2.2.0")


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "RWKV/rwkv-6-moe-11a41b"
_CONFIG_FOR_DOC = "Rwkv7MoeConfig"

def rwkv7_moe_linear_attention_cpu(receptance, key, value, time_decay, time_first, state):
    # For CPU fallback. Will be slower and probably take more memory than the custom CUDA kernel if not executed
    # within a torch.no_grad.
    batch, seq_length, _ = receptance.shape
    num_heads, head_size = time_first.shape
    key = key.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2).transpose(-2, -1)
    value = value.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2)
    receptance = receptance.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2)
    time_decay = torch.exp(-torch.exp(time_decay.float())).view(batch, seq_length, num_heads, head_size).permute(0, 2, 3, 1)
    time_first = time_first.float().reshape(-1, 1, 1).reshape(num_heads, -1, 1)
    out = torch.zeros_like(key).reshape(batch, seq_length, num_heads, head_size)

    for current_index in range(seq_length):
        current_receptance = receptance[:, :, current_index:current_index+1, :]
        current_key = key[:, :, :, current_index:current_index+1]
        current_value = value[:, :, current_index:current_index+1, :]
        current_time_decay = time_decay[:, :, :, current_index:current_index+1]
        attention_output = current_key @ current_value
        out[:, current_index] = (current_receptance @ (time_first * attention_output + state)).squeeze(2)
        with torch.no_grad():
            state = attention_output + current_time_decay * state

    return out, state

def rwkv7_moe_linear_attention(
    training,
    receptance,
    key,
    value,
    time_decay,
    time_first,
    state,
):
    no_cuda = any(t.device.type != "cuda" for t in [time_decay, time_first, receptance, key, value])
    # Launching the CUDA kernel for just one token will actually be slower (there is no for loop in the CPU version
    # in this case).
    one_token = key.size(1) == 1
    if not training or no_cuda or one_token:
        return rwkv7_moe_linear_attention_cpu(
            receptance, key, value, time_decay, time_first, state
        )
    else:
        batch, seq_length, _ = receptance.shape
        num_heads, head_size = time_first.shape
        key = key.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2) # B, T, H, K -> B, H, T, K
        value = value.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2) # B, T, H, K - > B, H, T, V
        receptance = receptance.float().view(batch, seq_length, num_heads, head_size).transpose(1, 2) # B, H, T, K
        time_decay = -torch.exp(time_decay.float()).view(batch, seq_length, num_heads, head_size).permute(0, 2, 1, 3) # B, T, H, K -> B, H, T, K
        time_first = time_first.float().reshape(num_heads, head_size) # H, K
        out, state = fused_recurrent_rwkv7(receptance, key, value, time_decay, time_first, scale=1.0, initial_state=state, output_final_state=True)
        return out.transpose(1, 2), state


class rwkv7MoeSelfAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hidden_size = config.hidden_size
        attention_hidden_size = config.attention_hidden_size
        self.attention_hidden_size = attention_hidden_size
        head_size = config.head_size
        num_heads = attention_hidden_size // head_size

        self.time_maa_x = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_w = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_k = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_v = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_r = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_g = nn.Parameter(torch.empty(1, 1, hidden_size))

        TIME_MIX_EXTRA_DIM = 32 # generate TIME_MIX for w,k,v,r,g
        if hidden_size == 4096: #7b
            TIME_MIX_EXTRA_DIM = 64
        self.time_maa_w1 = nn.Parameter(torch.empty(hidden_size, TIME_MIX_EXTRA_DIM*5))
        self.time_maa_w2 = nn.Parameter(torch.empty(5, TIME_MIX_EXTRA_DIM, hidden_size))

        self.time_decay = nn.Parameter(torch.empty(1, 1, attention_hidden_size))

        TIME_DECAY_EXTRA_DIM = 64
        if hidden_size == 4096: #7b
            TIME_DECAY_EXTRA_DIM = 128
        self.time_decay_w1 = nn.Parameter(torch.empty(hidden_size, TIME_DECAY_EXTRA_DIM))
        self.time_decay_w2 = nn.Parameter(torch.empty(TIME_DECAY_EXTRA_DIM, attention_hidden_size))

        self.time_faaaa = nn.Parameter(torch.empty(num_heads, config.head_size))

        
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.output = nn.Linear(attention_hidden_size, hidden_size, bias=False)
        self.ln_x = nn.GroupNorm(num_heads, hidden_size, eps=(1e-5)*(config.head_size_divisor**2))

    def extract_key_value(self, hidden, state=None):
        # Mix hidden with the previous timestep to produce key, value, receptance
        if hidden.size(1) == 1 and state is not None:
            shifted = state[0][:, :, self.layer_id]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[0][:, :, self.layer_id]
        if len(shifted.size()) == 2:
            shifted = shifted.unsqueeze(1)

        x = hidden

        B, T, C = hidden.shape

        xx = shifted - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        time_decay = x + xx * (self.time_maa_w + mw)
        key = x + xx * (self.time_maa_k + mk)
        value = x + xx * (self.time_maa_v + mv)
        receptance = x + xx * (self.time_maa_r + mr)
        gate = x + xx * (self.time_maa_g + mg)

        receptance = self.receptance(receptance)
        key = self.key(key)
        value = self.value(value)
        gate = F.silu(self.gate(gate))

        time_decay = torch.tanh(time_decay @ self.time_decay_w1) @ self.time_decay_w2
        time_decay = self.time_decay + time_decay

        if state is not None:
            state[0][:, :, self.layer_id] = hidden[:, -1]

        return receptance, key, value, gate, time_decay, state

    def forward(self, hidden, state=None, use_cache=False, seq_mode=True):
        receptance, key, value, gate, time_decay, state = self.extract_key_value(hidden, state=state)

        B,T,C = receptance.shape
        H, S = self.time_faaaa.shape

        layer_state = state[1][:, :, :, :, self.layer_id] if state is not None else None
        out, layer_state = rwkv7_moe_linear_attention(
            self.training, receptance, key, value, time_decay, self.time_faaaa, layer_state,
        )

        if layer_state is not None:
            state[1][:, :, :, :, self.layer_id] = layer_state

        out = out.reshape(B * T, H * S)
        out = F.group_norm(out, num_groups=H, weight=self.ln_x.weight.to(out.dtype), bias=self.ln_x.bias.to(out.dtype), eps=self.ln_x.eps).reshape(B, T, H * S)
        out = out.to(dtype=hidden.dtype) * gate
        out = self.output(out)
        return out, state

class rwkv7MoeFeedForwardExpert(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        hidden_size = config.hidden_size
        # https://github.com/BlinkDL/RWKV-LM/blob/3db37a72356b736966ddd377268f02b80963af3f/RWKV-v4neo/train.py#L168
        intermediate_size = (
            config.intermediate_size
            if config.intermediate_size is not None
            else int((config.hidden_size * 3.5) // 32 * 32)
        )

        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden, state=None):
        return self.value( torch.relu( self.key(hidden) ).square() )       

class rwkv7MoeFeedForward(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hidden_size = config.hidden_size

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.time_maa_k = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_r = nn.Parameter(torch.empty(1, 1, hidden_size))

        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.shared_expert = rwkv7MoeFeedForwardExpert(config, layer_id) if config.shared_expert else None
        self.experts = nn.ModuleList([rwkv7MoeFeedForwardExpert(config, layer_id) for _ in range(config.num_experts)])

        primes = [5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443]
        self.hash_prime = primes[layer_id]

    def forward(self, hidden, input_ids:torch.LongTensor, state=None):
        if hidden.size(1) == 1 and state is not None:
            shifted = state[2][:, :, self.layer_id]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[2][:, :, self.layer_id]
        if len(shifted.size()) == 2:
            shifted = shifted.unsqueeze(1)

        delta_hidden_to_shifted = shifted - hidden
        hidden_with_tokenshift = hidden + delta_hidden_to_shifted * self.time_maa_k
        receptance = hidden + delta_hidden_to_shifted * self.time_maa_r

        receptance = torch.sigmoid(self.receptance(receptance))

        # flatten batch and sequence dimensions of input_ids and hidden_with_tokenshift
        flat_input_ids = input_ids.flatten()
        flat_hidden_with_tokenshift = hidden_with_tokenshift.reshape(-1, hidden_with_tokenshift.size(-1))

        if self.shared_expert is not None:
            flat_value = self.shared_expert(flat_hidden_with_tokenshift)
        else:
            flat_value = torch.zeros_like(flat_hidden_with_tokenshift)

        # add in contributions from experts

        # find the expert index for each flat index (flattened batchseq index)
        expert_by_flat_idx = (flat_input_ids * self.hash_prime) % self.config.num_experts
        # one hot mask of expert choices by flat batchseq index
        expert_mask = torch.nn.functional.one_hot(expert_by_flat_idx, num_classes=self.config.num_experts).mT # expert_idx, flat_idx
        # go through each expert and add in their contributions
        for expert_idx in range(self.config.num_experts):
            expert = self.experts[expert_idx]
            # get a list of flat batchseq indices for the current expert
            flat_indices_for_expert = expert_mask[expert_idx].nonzero().flatten()
            if flat_indices_for_expert.size(-1) > 0:
                # select out the inputs from this expert's flat batchseq locations into a compact tensor
                expert_hidden_with_tokenshift = flat_hidden_with_tokenshift[flat_indices_for_expert]
                # run the compact tensor through the expert
                compact_expert_output = expert(expert_hidden_with_tokenshift)
                # add the expert's results to the appropriate original locations
                flat_value.index_add_(dim=0, index=flat_indices_for_expert, source=compact_expert_output)

        value = flat_value.view(hidden.size(0), hidden.size(1), hidden.size(2))

        if state is not None:
            state[2][:, :, self.layer_id] = hidden[:, -1]

        return receptance * value, state


class rwkv7MoeBlock(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if layer_id == 0:
            self.pre_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.attention = rwkv7MoeSelfAttention(config, layer_id)
        self.feed_forward = rwkv7MoeFeedForward(config, layer_id)

    def forward(self, hidden, input_ids:torch.LongTensor, state=None, use_cache=False, output_attentions=False, seq_mode=True):
        if self.layer_id == 0:
            hidden = self.pre_ln(hidden)
        attention, state = self.attention(self.ln1(hidden), state=state, use_cache=use_cache, seq_mode=seq_mode)
        hidden = hidden + attention

        feed_forward, state = self.feed_forward(self.ln2(hidden), input_ids=input_ids, state=state)
        hidden = hidden + feed_forward

        outputs = (hidden, state)
        if output_attentions:
            outputs += (attention,)
        else:
            outputs += (None,)

        return outputs


class rwkv7MoePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = rwkv7MoeConfig
    base_model_prefix = "rwkv7moe"
    _no_split_modules = ["rwkv7MoeBlock"]
    _keep_in_fp32_modules = ["time_decay", "time_first"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, rwkv7MoeSelfAttention):
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size
            attention_hidden_size = module.attention_hidden_size
            head_size = module.config.head_size
            num_heads = attention_hidden_size // head_size

            ratio_0_to_1 = layer_id / (num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0

            time_weight = torch.tensor(
                [i / hidden_size for i in range(hidden_size)],
                dtype=module.time_maa_k.dtype,
                device=module.time_maa_k.device,
            )
            time_weight = time_weight[None, None, :]

            decay_speed = [
                -6.0 + 5.0 * (h / (attention_hidden_size - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                for h in range(attention_hidden_size)
            ]
            decay_speed = torch.tensor(decay_speed, dtype=module.time_decay.dtype, device=module.time_decay.device)
            tmp = torch.tensor(
                [
                    (1.0 - (i / (attention_hidden_size - 1.0))) * ratio_0_to_1 + 0.1 * ((i + 1) % 3 - 1)
                    for i in range(attention_hidden_size)
                ],
                dtype=module.time_faaaa.dtype,
                device=module.time_faaaa.device,
            )

            with torch.no_grad():
                module.time_maa_x.data = 1.0 - torch.pow(time_weight, ratio_1_to_almost0)
                module.time_maa_w.data = 1.0 - torch.pow(time_weight, ratio_1_to_almost0)
                module.time_maa_k.data = 1.0 - torch.pow(time_weight, ratio_1_to_almost0)
                module.time_maa_v.data = 1.0 - (torch.pow(time_weight, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                module.time_maa_r.data = 1.0 - torch.pow(time_weight, 0.5 * ratio_1_to_almost0)
                module.time_maa_g.data = 1.0 - torch.pow(time_weight, 0.5 * ratio_1_to_almost0)

                TIME_MIX_EXTRA_DIM = 32 # generate TIME_MIX for w,k,v,r,g
                module.time_maa_w1.data = torch.zeros(hidden_size, TIME_MIX_EXTRA_DIM*5, dtype=module.time_maa_w1.dtype, device=module.time_maa_w1.device).uniform_(-1e-4, 1e-4)
                module.time_maa_w2.data = torch.zeros(5, TIME_MIX_EXTRA_DIM, hidden_size, dtype=module.time_maa_w2.dtype, device=module.time_maa_w2.device).uniform_(-1e-4, 1e-4)

                TIME_DECAY_EXTRA_DIM = 64
                module.time_decay_w1.data = torch.zeros(hidden_size, TIME_DECAY_EXTRA_DIM, dtype=module.time_decay_w1.dtype, device=module.time_decay_w1.device).uniform_(-1e-4, 1e-4)
                module.time_decay_w2.data = torch.zeros(TIME_DECAY_EXTRA_DIM, attention_hidden_size, dtype=module.time_decay_w2.dtype, device=module.time_decay_w2.device).uniform_(-1e-4, 1e-4)

                module.time_decay.data = decay_speed.reshape(num_heads, head_size)
                module.time_faaaa.data = tmp.reshape(num_heads, head_size)

        elif isinstance(module, rwkv7MoeFeedForward):
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size

            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0

            time_weight = torch.tensor(
                [i / hidden_size for i in range(hidden_size)],
                dtype=module.time_maa_k.dtype,
                device=module.time_maa_k.device,
            )
            time_weight = time_weight[None, None, :]

            with torch.no_grad():
                module.time_maa_k.data = 1.0 - torch.pow(time_weight, ratio_1_to_almost0)
                module.time_maa_r.data = 1.0 - torch.pow(time_weight, ratio_1_to_almost0)


@dataclass
class rwkv7MoeOutput(ModelOutput):
    """
    Class for the RWKV model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    state: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class rwkv7MoeCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    state: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


rwkv7MOE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.) This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config ([`rwkv7MoeConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

rwkv7MOE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary. If `past_key_values` is used, only `input_ids` that do not have their
            past calculated should be passed as `input_ids`. Indices can be obtained using [`AutoTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input
            IDs?](../glossary#input-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        state (tuple of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        use_cache (`bool`, *optional*):
            If set to `True`, the last state is returned and can be used to quickly generate the next logits.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare rwkv7Moe Model transformer outputting raw hidden-states without any specific head on top.",
    rwkv7MOE_START_DOCSTRING,
)
class rwkv7MoeModel(rwkv7MoePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([rwkv7MoeBlock(config, layer_id=idx) for idx in range(config.num_hidden_layers)])
        self.ln_out = nn.LayerNorm(config.hidden_size)

        self.layers_are_rescaled = False
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(rwkv7MOE_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=rwkv7MoeOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, rwkv7MoeOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # FIXME - training is supportable with the CUDA code
        # rwkv7 only support inference in huggingface.
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training == self.layers_are_rescaled and (
            self.embeddings.weight.dtype == torch.float16 or self.embeddings.weight.dtype == torch.bfloat16
        ):
            self._rescale_layers()

        if input_ids is None:
            raise ValueError("RWKV-MoE requires that you specify input_ids, as it uses these to select experts")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if state is None:
            state = []
            head_size = self.config.head_size
            num_heads = self.config.attention_hidden_size // head_size
            state_attn_x = torch.zeros(
                    (inputs_embeds.size(0), self.config.hidden_size, self.config.num_hidden_layers),
                    dtype=inputs_embeds.dtype,
                    requires_grad=False,
                    device=inputs_embeds.device,
                ).contiguous()
            state_attn_kv = torch.zeros(
                    (
                        inputs_embeds.size(0),
                        num_heads,
                        head_size,
                        head_size,
                        self.config.num_hidden_layers,
                    ),
                    dtype=torch.float32,
                    requires_grad=False,
                    device=inputs_embeds.device,
                ).contiguous()
            state_ffn_x = torch.zeros(
                    (inputs_embeds.size(0), self.config.hidden_size, self.config.num_hidden_layers),
                    dtype=inputs_embeds.dtype,
                    requires_grad=False,
                    device=inputs_embeds.device,
                ).contiguous()
            state.append(state_attn_x)
            state.append(state_attn_kv)
            state.append(state_ffn_x)

        seq_mode = inputs_embeds.shape[1] > 1
        hidden_states = inputs_embeds

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for idx, block in enumerate(self.blocks):
            hidden_states, state, attentions = block(
                hidden_states, input_ids=input_ids, state=state, use_cache=use_cache, output_attentions=output_attentions, seq_mode=seq_mode
            )
            if (
                self.layers_are_rescaled
                and self.config.rescale_every > 0
                and (idx + 1) % self.config.rescale_every == 0
            ):
                hidden_states = hidden_states / 2

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if output_attentions:
                all_self_attentions = all_self_attentions + (attentions,)

        hidden_states = self.ln_out(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return (hidden_states, state, all_hidden_states, all_self_attentions)

        return rwkv7MoeOutput(
            last_hidden_state=hidden_states,
            state=state,
            hidden_states=all_hidden_states,  # None
            attentions=all_self_attentions,  # None
        )

    def _rescale_layers(self):
        # Layers should be rescaled for inference only.
        if self.layers_are_rescaled == (not self.training):
            return
        if self.config.rescale_every > 0:
            with torch.no_grad():
                for block_id, block in enumerate(self.blocks):
                    if self.training:
                        block.attention.output.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                        block.feed_forward.shared_expert.value.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                        for expert in block.feed_forward.experts:
                            expert.value.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                    else:
                        # Deal with quantization statistics
                        if hasattr(block.attention.output.weight, "SCB"):
                            block.attention.output.weight.SCB.div_(2 ** int(block_id // self.config.rescale_every))
                            block.feed_forward.shared_expert.value.weight.SCB.div_(2 ** int(block_id // self.config.rescale_every))
                            for expert in block.feed_forward.experts:
                                expert.value.weight.SCB.div_(2 ** int(block_id // self.config.rescale_every))
                        elif hasattr(block.attention.output.weight, "quant_state"):
                            self._bnb_4bit_dequantize_and_rescale(block.attention.output, block_id)
                            self._bnb_4bit_dequantize_and_rescale(block.feed_forward.shared_expert.value, block_id)
                            for expert in block.feed_forward.experts:
                                self._bnb_4bit_dequantize_and_rescale(expert.value, block_id)
                        else:
                            block.attention.output.weight.div_(2 ** int(block_id // self.config.rescale_every))
                            block.feed_forward.shared_expert.value.weight.div_(2 ** int(block_id // self.config.rescale_every))
                            for expert in block.feed_forward.experts:
                                expert.value.weight.div_(2 ** int(block_id // self.config.rescale_every))

        self.layers_are_rescaled = not self.training

    def _bnb_4bit_dequantize_and_rescale(self, target_layer, block_id):
        r"""
        Perform the dequantization and rescaling of the weights of a given layer. After that operation the layer will
        be quantized again.
        """
        if not is_bitsandbytes_available():
            raise ImportError("Please install bitsandbytes to use this method.")
        import bitsandbytes as bnb

        dequant_weights = bnb.functional.dequantize_4bit(target_layer.weight.data, target_layer.weight.quant_state)

        dequant_weights.div_(2 ** int(block_id // self.config.rescale_every))

        # re-quantize the model:
        # we need to put it first on CPU then back to the device
        # this will create an overhead :/
        # We set requires_grad=False as we cannot compute gradients on top of 4bit parameters anyway and to avoid
        # bugs with bnb
        quant_weight = bnb.nn.Params4bit(dequant_weights.to("cpu"), requires_grad=False).to(dequant_weights.device)
        setattr(target_layer, "weight", quant_weight)


# copied from HuggingFace https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py
@add_start_docstrings(
    """
    The rwkv7Moe Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    rwkv7MOE_START_DOCSTRING,
)
class rwkv7MoeForCausalLM(rwkv7MoePreTrainedModel):
    _tied_weights_keys = ["head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = rwkv7MoeModel(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, state=None, inputs_embeds=None, **kwargs):
        # only last token for inputs_ids if the state is passed along.
        if state is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and state is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["state"] = state
        return model_inputs

    @add_start_docstrings_to_model_forward(rwkv7MOE_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=rwkv7MoeCausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, rwkv7MoeCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        logits = self.head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return rwkv7MoeCausalLMOutput(
            loss=loss,
            logits=logits,
            state=outputs.state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
