import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from torch import Tensor
from typing import Optional,Tuple
from transformers.activations import ACT2FN
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.configuration_utils import PretrainedConfig

from .modeling_pos import (
    get_abs_pos, 
    get_1d_sincos_pos_embed_from_grid, 
    get_2d_sincos_pos_embed_from_grid, 
    get_2d_sincos_pos_embed
)


class VisualMLP(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.act = nn.SiLU()

        self.w1 = nn.Linear(hidden_size,config.intermediate_size)
        self.w2 = nn.Linear(config.intermediate_size, hidden_size)
        self.w3 = nn.Linear(hidden_size, config.intermediate_size)
        self.ffn_ln = nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps)

    def forward(self,hidden_states: Tensor) -> Tensor:
        hidden_states = self.act(self.w1(hidden_states)) * self.w3(hidden_states)
        hidden_states = self.ffn_ln(hidden_states)
        hidden_states = self.w2(hidden_states)
        return hidden_states

class VisualMultiHeadAttention(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.save_attention = False

        grids = config.grid_size
        self.register_buffer(
            'q_pos_embed', 
            torch.from_numpy(get_1d_sincos_pos_embed_from_grid(config.hidden_size, np.arange(config.num_learnable_queries, dtype=np.float32))).float()
        )
        self.register_buffer(
            'k_pos_embed', 
            torch.from_numpy(get_2d_sincos_pos_embed(config.hidden_size, grids, cls_token=True)).float()
        )
        

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        
        qk_pos_embed = torch.cat([self.q_pos_embed, self.k_pos_embed], dim = 0).unsqueeze(0).to(dtype=hidden_states.dtype)
        
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states + qk_pos_embed))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        attention_mask = encoder_attention_mask

        mixed_query_layer = self.query(hidden_states + self.q_pos_embed.unsqueeze(0).to(dtype=hidden_states.dtype))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        outputs = outputs + (past_key_value,)
        return outputs
