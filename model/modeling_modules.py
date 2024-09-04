import torch
import numpy as np
import torch.nn as nn
from .modeling_transformer import VisionEncoder, MLP, DownSampleMLP
from transformers.configuration_utils import PretrainedConfig

class scene_level(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        self.embed_2d = nn.Linear(config.dim_2d,config.hidden_size)
        self.embed_3d = nn.Linear(config.dim_3d,config.hidden_size)
        self.downsample = DownSampleMLP(2*config.hidden_size,config.hidden_size,config)
        self.vision_encoder = VisionEncoder(config)
        super().__init__(config)

    def forward(
        self,
        input_2d: torch.Tensor,
        input_3d: torch.Tensor,
    ) -> torch.Tensor:

        embedding_2d = self.embed_2d(input_2d)
        embedding_3d = self.embed_3d(input_3d)

        input_embedding = self.downsample(torch.cat((embedding_2d,embedding_3d),dim=-1))
        vision_output = self.vision_encoder(input_embedding)['last_hidden_state']
        return vision_output
    
class object_level(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        self.embed_object = nn.Linear(config.dim_object,config.hidden_size)
        self.
        super().__init__(config)

    def forward(
        self,
        input_object: torch.Tensor,
        input_scene: torch.Tensor,
    ) -> torch.Tensor:
        