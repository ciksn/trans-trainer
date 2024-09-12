import torch
import numpy as np
from transformers.configuration_utils import PretrainedConfig
from transformers import CLIPConfig,LlamaConfig,CLIPModel,LlamaForCausalLM,LlamaTokenizer
from .configuration_model import MAINconfig,MAINMultiTaskConfig,MAINAbstractorConfig
from .modeling import MAIN
from .modeling_abstractor import MAINVisualAbstractorModel

from icecream import ic

def build_model(model_args, model_config: MAINconfig):
    # for training time
    model = MAIN(model_config)
    model.language_model = LlamaForCausalLM.from_pretrained(model_args.language_model, config=model_config.language_model_config)
    model.visual_backbone = CLIPModel.from_pretrained(model_args.visual_backbone, config=model_config.visual_backbone_config).vision_model
    return model

def build_tokenizer(model_args):
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path,
        add_eos_token = True
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def build_config(model_args):
    config = MAINconfig(
        64,
        2,
        None,
        None,
        None,
        None,
    )
    config.visual_backbone_config = CLIPConfig.from_pretrained(model_args.visual_backbone)
    config.language_model_config = LlamaConfig.from_pretrained(model_args.language_model)
    return config

if __name__=="__main__":
    pass