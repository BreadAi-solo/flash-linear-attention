# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.rwkv7moe.configuration_rwkv7moe import RWKV7MOEConfig
from fla.models.rwkv7moe.modeling_rwkv7moe import RWKV7MOEForCausalLM, RWKV7MOEModel

AutoConfig.register(RWKV7MOEConfig.model_type, RWKV7MOEConfig, True)
AutoModel.register(RWKV7MOEConfig, RWKV7MOEModel, True)
AutoModelForCausalLM.register(RWKV7MOEConfig, RWKV7MOEForCausalLM, True)


__all__ = ['RWKV7MOEConfig', 'RWKV7MOEForCausalLM', 'RWKV7MOEModel']
