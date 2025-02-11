# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.rwkv7moe.configuration_rwkv7moe import rwkv7MoeConfig
from fla.models.rwkv7moe.modeling_rwkv7moe import rwkv7MoeForCausalLM, RWKV7MOEModel

AutoConfig.register(rwkv7MoeConfig.model_type, RWKV7MOEConfig, True)
AutoModel.register(rwkv7MoeConfig, RWKV7MOEModel, True)
AutoModelForCausalLM.register(rwkv7MoeConfig, rwkv7MoeForCausalLM, True)


__all__ = ['rwkv7MoeConfig', 'rwkv7MoeForCausalLM', 'RWKV7MOEModel']
