import torch
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

class custom_model(PreTrainedModel):
    """
    Model definition here
    """
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super(custom_model,self).__init__(config, *inputs, **kwargs)
        self.post_init()

    def _init_weights(self, module):
        module.reset_parameters()


    