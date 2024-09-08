import torch
import numpy as np
import torch.nn as nn
from typing import Tuple,Dict,Optional
from transformers.configuration_utils import PretrainedConfig
from model.modeling_transformer import TransformerEncoder, TransformerDecoder, TransformerTextEncoder

from icecream import ic

class SceneLevel(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.embed_2d = nn.Linear(config.dim_2d,config.hidden_size)
        self.embed_3d = nn.Linear(config.dim_3d,config.hidden_size)
        self.downsample = nn.Linear(2*config.hidden_size,config.hidden_size)
        self.vision_encoder = TransformerEncoder(config)

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
        super().__init__()
        self.embed_object = nn.Linear(config.dim_object,config.hidden_size)
        self.object_encoder = TransformerEncoder(config)
        self.object_decoder = TransformerDecoder(config)
        self.learnable_parameters = nn.Parameter(torch.randn(1, config.num_learnable_queries, config.hidden_size))

    def forward(
        self,
        input_object: torch.Tensor,
        input_scene: torch.Tensor,
    ) -> torch.Tensor:
        mixed_query_embed = self.object_encoder(self.embed_object(input_object))[0]
        encoder_hidden_states = input_scene + self.learnable_parameters # do not move here
        output = self.object_decoder(mixed_query_embed,encoder_hidden_states)[0]
        return output
    
class MotionLevel(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.vision_encoder = TransformerEncoder(config)
        # self.cross_attn = MultiHeadAttention(config)
        self.cross_attn = TransformerDecoder(config)
        self.learnable_parameters = nn.Parameter(torch.randn(1, config.num_learnable_queries, config.hidden_size))

    def forward(
        self,
        input_scene: torch.Tensor,
        input_object: torch.Tensor,
    ) -> torch.Tensor:
        actual_input = torch.cat((input_scene,input_object),dim=1)
        output = self.vision_encoder(actual_input)[0]
        output = self.cross_attn(self.learnable_parameters,output)[0]
        return output
    
class TextGeneration(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(config.vocab_size,config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.action_encoder = TransformerTextEncoder(config)
        self.reason_encoder = TransformerTextEncoder(config)

    def forward(
        self,
        input_visual: torch.Tensor,
        action_input_ids: torch.Tensor,
        reason_input_ids: torch.Tensor,
        action_casual_mask: Optional[torch.Tensor] = None,
        action_attention_mask: Optional[torch.Tensor] = None,
        reason_casual_mask: Optional[torch.Tensor] = None,
        reason_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        B,T,H = input_visual.size()
        action_word_embeds = self.word_embedding(action_input_ids)
        reason_word_embeds = self.word_embedding(reason_input_ids)
        action_input = torch.cat((input_visual, action_word_embeds),dim=1).to(input_visual.device)
        reason_input = torch.cat((input_visual, reason_word_embeds),dim=1).to(input_visual.device)
        probs_action = self.action_encoder(action_input, action_casual_mask)[0][:,T:,:]
        probs_reason = self.reason_encoder(reason_input, reason_casual_mask)[0][:,T:,:]

        logits_action = self.lm_head(probs_action)
        logits_reason = self.lm_head(probs_reason)

        return logits_action, logits_reason
    