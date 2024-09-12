import torch
import numpy as np
from transformers.configuration_utils import PretrainedConfig
from transformers import CLIPConfig,LlamaConfig,CLIPModel,LlamaForCausalLM,LlamaTokenizer
from .configuration_model import MAINconfig,MAINMultiTaskConfig,MAINAbstractorConfig
from .modeling import MAIN
from .modeling_abstractor import MAINVisualAbstractorModel

from icecream import ic

def build_model(model_args, model_config: MAINconfig, tokenizer):
    clip_model = CLIPModel.from_pretrained(model_args.visual_backbone, config=model_config.visual_backbone_config)
    abstractor = MAINVisualAbstractorModel(model_config.visual_abstractor_config, model_config.language_model_config.hidden_size)
    language_model = LlamaForCausalLM.from_pretrained(model_args.language_model, config=model_config.language_model_config)
    model = MAIN(clip_model.vision_model, abstractor, None, language_model, model_config,tokenizer)
    return model

def build_tokenizer(model_args):
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path,
        add_eos_token = True
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def build_config(model_args):
    visual_backbone_config = CLIPConfig.from_pretrained(model_args.visual_backbone)
    visual_abstractor_config = MAINAbstractorConfig()
    multi_task_config = MAINMultiTaskConfig()
    llama_config = LlamaConfig.from_pretrained(model_args.language_model)
    config = MAINconfig(64,visual_backbone_config,visual_abstractor_config,multi_task_config,llama_config)
    return config