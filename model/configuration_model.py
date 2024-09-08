from os import PathLike
from typing import Any, Dict
from transformers.configuration_utils import PretrainedConfig

class custom_model_config(PretrainedConfig):
    """
    Defination here
    """
    model_type = "custom_model" # model name

    def __init__(self,
        vocab_size = 3000,
        hidden_size = 1024,
        encoder_hidden_size = 1024,
        intermediate_size = 2048,
        num_hidden_layers = 2,
        num_decoder_hidden_layers = 4,
        num_attention_heads = 32,
        dim_2d = 768,
        dim_3d = 1024,
        dim_object = 2048,
        caption_seq_len = 40,
        video_seq_len = 32,
        num_learnable_queries = 32,
        dropout_rate = 0.3,
        layer_norm_eps = 1e-6,
        label_smoothing = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_decoder_hidden_layers = num_decoder_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dim_2d = dim_2d
        self.dim_3d = dim_3d
        self.dim_object = dim_object
        self.caption_seq_len = caption_seq_len
        self.video_seq_len = video_seq_len
        self.num_learnable_queries = num_learnable_queries
        self.dropout_rate = dropout_rate
        self.layer_norm_eps = layer_norm_eps
        self.label_smoothing = label_smoothing
        super().__init__(**kwargs)

if __name__ == "__main__":
    custom_model_config_a = custom_model_config()
    custom_model_config_a.save_pretrained("/home/zeyu/work/deep_learning/functional_files/trans_trainer/test")
    # custom_model_config_b = custom_model_config.from_pretrained("/home/zeyu/work/deep_learning/functional_files/trans-trainer/test/config.json")
    print(custom_model_config_a.to_dict())