import torch
import numpy as np
import torch.nn as nn
from .modeling_transformer import TransformerEncoder, TransformerDecoder, MLP, DownSampleMLP
from transformers.configuration_utils import PretrainedConfig

class SceneLevel(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        self.embed_2d = nn.Linear(config.dim_2d,config.hidden_size)
        self.embed_3d = nn.Linear(config.dim_3d,config.hidden_size)
        self.downsample = DownSampleMLP(2*config.hidden_size,config.hidden_size,config)
        self.vision_encoder = TransformerEncoder(config)
        super().__init__(config)

    def forward(
        self,
        input_2d: torch.Tensor,
        input_3d: torch.Tensor,
    ) -> torch.Tensor:

        embedding_2d = self.embed_2d(input_2d)
        embedding_3d = self.embed_3d(input_3d)

        input_embedding = self.downsample(torch.cat((embedding_2d,embedding_3d),dim=-1))
        vision_output = self.vision_encoder(input_embedding)[0]
        return vision_output
    
class ObjectLevel(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        self.embed_object = nn.Linear(config.dim_object,config.hidden_size)
        self.object_encoder = TransformerEncoder(config)
        self.object_decoder = TransformerDecoder(config)
        self.learnable_parameters = nn.Parameter(torch.randn(1, config.num_learnable_queries, config.hidden_size))
        super().__init__(config)

    def forward(
        self,
        input_object: torch.Tensor,
        input_scene: torch.Tensor,
    ) -> torch.Tensor:
        encoder_hidden_states = self.object_encoder(input_object)[0]
        mixed_query_embed = input_scene + self.learnable_parameters
        output = self.object_decoder(encoder_hidden_states,mixed_query_embed)[0]
        return output
    
class MotionLevel(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        self.vision_encoder = TransformerEncoder(config)
        super().__init__(config)

    def generate_binary_mask(self,):
        pass

    def forward(
        self,
        input_scene: torch.Tensor,
        input_object: torch.Tensor,
    ) -> torch.Tensor:
        actual_input = torch.cat((input_scene,input_object),dim=1)
        output = self.vision_encoder.forward(actual_input)
        return output
    
class TextGeneration(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        self.caption_seq_len = config.caption_seq_len
        self.word_embedding = nn.Embedding(config.vocab_size,config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        super().__init__(config)

    def forward(
        self,
        input_visual: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # TODO
        pass

    def generate(self,input_visual):
        pass
    