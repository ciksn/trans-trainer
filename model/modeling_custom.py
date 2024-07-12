from typing import Tuple,Dict
import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput

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

    def get_tokenizer(self,) -> PreTrainedTokenizer:
        return self.tokenizer
        
    def _prepare_model_inputs(self, inputs: torch.Tensor | None = None, bos_token_id: int | None = None, model_kwargs: torch.Dict[str, torch.Tensor] | None = None) -> Tuple[Tensor, str | None, Dict[str, Tensor]]:
        return super()._prepare_model_inputs(inputs, bos_token_id, model_kwargs)

    def forward(self,vid_feat,input_ids,attention_mask,):
        """
        The inputs need to be flattened here since (**input) when called
        The output of the main model should be "ModelOutput"
        or dict contains key "loss" / 1st element be loss
        """
        loss = None
        #when computing loss, remember to distinct single batch when training or batch of list when eval
        
        return ModelOutput(
            loss = loss
        )