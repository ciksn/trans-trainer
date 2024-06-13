from transformers.configuration_utils import PretrainedConfig

class custom_model_config(PretrainedConfig):
    """
    Defination here
    """
    
    model_type = "custom_model" # model name

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
