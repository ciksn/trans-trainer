from typing import Tuple,Dict
import torch
from torch import Tensor
from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput

class custom_model(PreTrainedModel):
    """
    Model definition here
    """
    def __init__(self, config: PretrainedConfig, tokenizer = None, *inputs, **kwargs):
        super(custom_model,self).__init__(config, *inputs, **kwargs)
        self.tokenizer = tokenizer
        self.post_init()

    def _init_weights(self, module):
        module.reset_parameters()

    def get_tokenizer(self,) -> PreTrainedTokenizer:
        return self.tokenizer
        
    def _prepare_model_inputs(self, inputs: torch.Tensor | None = None, bos_token_id: int | None = None, model_kwargs: torch.Dict[str, torch.Tensor] | None = None) -> Tuple[Tensor, str | None, Dict[str, Tensor]]:
        return super()._prepare_model_inputs(inputs, bos_token_id, model_kwargs)

    def forward(self,inputs):
        """
        The output of the main model should be "ModelOutput"
        or dict contains key "loss" / 1st element be loss
        """

        return ModelOutput(
            
        )