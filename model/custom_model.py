import torch
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput

class custom_model(PreTrainedModel):
    """
    Model definition here
    """
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super(custom_model,self).__init__(config, *inputs, **kwargs)

        self.post_init()

    def _init_weights(self, module):
        module.reset_parameters()

    def forward():
        """
        The output of the main model should be "ModelOutput"
        or dict contains key "loss" / 1st element be loss
        """
        return ModelOutput(
            
        )