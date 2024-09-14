import os
import copy
from os import PathLike
from typing import Any, Dict, Union, Optional
from transformers.configuration_utils import PretrainedConfig
from transformers import CLIPConfig, LlamaConfig
from icecream import ic

class MAINAbstractorConfig(PretrainedConfig):
    model_type = "MAIN_visual_abstract"

    def __init__(
        self,
        hidden_size=1024,  #
        num_hidden_layers=6,  #
        num_attention_heads=16,  #
        intermediate_size=4096,  #
        attention_probs_dropout_prob=0.1,  #
        initializer_range=0.02,
        layer_norm_eps=1e-6,  #
        encoder_hidden_size=1024,  #
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.encoder_hidden_size = encoder_hidden_size

class MAINMultiTaskConfig(PretrainedConfig):
    r"""
    """

    model_type = "MAIN_multi_task"

    def __init__(
        self,
        hidden_size=1024,  #
        num_hidden_layers=6,  #
        num_attention_heads=16,  #
        intermediate_size=4096,  #
        attention_probs_dropout_prob=0.1,  #
        initializer_range=0.02,
        layer_norm_eps=1e-6,  #
        encoder_hidden_size=1024,  #
        detection_size=4,
        num_query_tokens=32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.encoder_hidden_size = encoder_hidden_size
        self.detection_size = detection_size
        self.num_query_tokens = num_query_tokens

class MAINconfig(PretrainedConfig):
    """
    Defination here
    """
    model_type = "MAIN" # model name
    is_composition = True

    def __init__(self,
        num_query_tokens: int = 64,
        bos_token_id = 1,
        pad_token_id = 2,
        visual_backbone_config = None,
        visual_abstractor_config = None,
        multi_task_config = None,
        language_model_config = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_query_tokens = num_query_tokens
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id

        if visual_backbone_config is None:
            visual_backbone_config = CLIPConfig().to_dict()
        
        if visual_abstractor_config is None:
            visual_abstractor_config = MAINAbstractorConfig().to_dict()

        if multi_task_config is None:
            multi_task_config = MAINMultiTaskConfig().to_dict()

        if language_model_config is None:
            language_model_config = LlamaConfig(pad_token_id=2).to_dict()

        self.visual_backbone_config =  CLIPConfig(**visual_backbone_config)
        self.visual_abstractor_config = MAINAbstractorConfig(**visual_abstractor_config)
        self.multi_task_config = MAINMultiTaskConfig(**multi_task_config)
        self.language_model_config = LlamaConfig(**language_model_config)

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        output["model_type"] = self.__class__.model_type
        output['visual_backbone_config'] = self.visual_backbone_config.to_dict()
        output['visual_abstractor_config'] = self.visual_abstractor_config.to_dict()
        output['multi_task_config'] = self.multi_task_config.to_dict()
        output['language_model_config'] = self.language_model_config.to_dict()
        return output

if __name__ == "__main__":
    a = LlamaConfig().to_dict()
    print(a)