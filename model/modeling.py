from typing import Tuple,Dict
import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from .modeling_transformer import DownSampleMLP
from .modeling_modules import SceneLevel, ObjectLevel, MotionLevel, TextGeneration

class custom_model(PreTrainedModel):
    """
    Model definition here

    Input:
        Contains input data, label
    Output:
        Inherit from ModelOutput -> the first element must be loss

    """
    def __init__(self, config: PretrainedConfig, tokenizer: PreTrainedTokenizer | None = None, *inputs, **kwargs):
        super(custom_model,self).__init__(config, *inputs, **kwargs)
        self.config = config
        self.tokenizer = tokenizer
        self.loss = nn.CrossEntropyLoss()
        self.SceneLevel = SceneLevel(config)
        self.ObjectLevel = ObjectLevel(config)
        self.MotionLevel = MotionLevel(config)

        self.module_downsample = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.input_downsample = nn.Linear(config.dim_2d + config.dim_3d + config.dim_object, config.hidden_size)
        self.mixed_downsample = DownSampleMLP(2 * config.hidden_size, config.hidden_size)

        self.text_generation = TextGeneration(config)

    def get_tokenizer(self,) -> PreTrainedTokenizer:
        return self.tokenizer
        
    def _prepare_model_inputs(self, inputs: torch.Tensor | None = None, bos_token_id: int | None = None, model_kwargs: torch.Dict[str, torch.Tensor] | None = None) -> Tuple[Tensor, str | None, Dict[str, Tensor]]:
        return super()._prepare_model_inputs(inputs, bos_token_id, model_kwargs)

    def forward(self,input_2d, input_3d, input_object, labels):
        """
        The inputs need to be flattened here since (**input) when called
        The output of the main model should be "ModelOutput"
        or dict contains key "loss" / 1st element be loss
        """
        scene_output = self.SceneLevel(input_2d, input_3d)
        object_output = self.ObjectLevel(input_object, scene_output)
        motion_output = self.MotionLevel(scene_output,object_output)

        mixed_output = self.module_downsample(torch.cat((scene_output,object_output,motion_output),dim=-1))
        mixed_feat = self.input_downsample(torch.cat((input_2d,input_3d,input_object),dim=-1))
        actual_visual_input = self.mixed_downsample(torch.cat((mixed_output,mixed_feat),dim=-1))

        #TODO below
        input_ids, attention_mask = self.tokenizer(labels,...)
        logits = self.text_generation(actual_visual_input,input_ids,attention_mask)

        #TODO Trick: consider label smoothing here
        loss = self.loss.forward(logits, self.tokenizer(labels,...))
        
        #when computing loss, remember to distinct single batch when training or batch of list when eval

        return ModelOutput(
            loss = loss,
            logits = logits
        )

    def generate(self,input_2d, input_3d, input_object):
        pass
