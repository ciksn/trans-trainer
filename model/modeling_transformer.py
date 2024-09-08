import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch import Tensor
from typing import Optional,Tuple,Union
from transformers.activations import ACT2FN
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput

from model.modeling_pos import (
    get_abs_pos,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed
)

from icecream import ic

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = QuickGELU()
        self.drop_out = nn.Dropout(config.dropout_rate)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.drop_out(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class VisualMLP(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
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
    
class MultiHeadAttention(nn.Module):
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

        self.dropout = nn.Dropout(config.dropout_rate)
        self.save_attention = False

        # grids = config.grid_size
        # self.register_buffer(
        #     'q_pos_embed', 
        #     torch.from_numpy(get_1d_sincos_pos_embed_from_grid(config.hidden_size, np.arange(config.num_learnable_queries, dtype=np.float32))).float()
        # )
        # self.register_buffer(
        #     'k_pos_embed', 
        #     torch.from_numpy(get_2d_sincos_pos_embed(config.hidden_size, grids, cls_token=False)).float()
        # )

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
        q_hidden_states,
        kv_hidden_states=None,
        attention_mask=None,
        head_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        """
        Args:
            hidden_state: indicate query
            attention_mask: final attention weights mask
                range [-inf,0] size [batch_size, q_len, k_len]
            head_mask: attention head mask to mask several given heads
                range [0,1](binary) size [batch_size,v_len]
            encoder_hidden_state: input for key-value
            encoder_attention_mask: mask encoder input
            past_key_value: during generation
            output_attention: bool indicate whether output attention weights 
        """
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.

        if kv_hidden_states is None:
            kv_hidden_states = q_hidden_states

        q_pos_embed = torch.from_numpy(get_1d_sincos_pos_embed_from_grid(self.config.hidden_size,np.arange(q_hidden_states.size(1),dtype=np.float32))).float().to(q_hidden_states.device)
        # k_pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.config.hidden_size, hidden_states.size(1), cls_token=False)).float()
        k_pos_embed = torch.from_numpy(get_1d_sincos_pos_embed_from_grid(self.config.hidden_size,np.arange(kv_hidden_states.size(1),dtype=np.float32))).float().to(q_hidden_states.device)

        # TODO what qk_pos_embed here for
        qk_pos_embed = torch.cat([q_pos_embed, k_pos_embed], dim = 0).unsqueeze(0).to(dtype=q_hidden_states.dtype)
        
        key_layer = self.transpose_for_scores(self.key(kv_hidden_states + k_pos_embed))
        value_layer = self.transpose_for_scores(self.value(kv_hidden_states))

        mixed_query_layer = self.query(q_hidden_states + q_pos_embed.unsqueeze(0).to(dtype=q_hidden_states.dtype))

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

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MultiHeadAttention(config)
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            q_hidden_states=hidden_states,
            kv_hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MultiHeadAttention(config)
        self.self_attn_dropout = nn.Dropout(config.dropout_rate)
        self.cross_attn = MultiHeadAttention(config)
        self.cross_attn_dropout = nn.Dropout(config.dropout_rate)
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.cross_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.mlp_dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        q_hidden_states: torch.Tensor,
        kv_hidden_states: torch.Tensor,
        q_attention_mask: Optional[torch.Tensor] = None,
        kv_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = q_hidden_states

        q_hidden_states = self.input_layernorm(q_hidden_states)
        q_hidden_states, self_attn_weights = self.self_attn.forward(
            q_hidden_states=q_hidden_states,
            kv_hidden_states=q_hidden_states,
            attention_mask=q_attention_mask,
            output_attentions=output_attentions,
        )
        q_hidden_states = self.self_attn_dropout(q_hidden_states)
        q_hidden_states = q_hidden_states + residual
        residual = q_hidden_states

        q_hidden_states = self.cross_layernorm(q_hidden_states)
        q_hidden_states, cross_attn_weights = self.cross_attn.forward(
            q_hidden_states=q_hidden_states,
            kv_hidden_states=kv_hidden_states,
            attention_mask=kv_attention_mask,
            output_attentions=output_attentions,
        )

        q_hidden_states = self.cross_attn_dropout(q_hidden_states)
        q_hidden_states = q_hidden_states + residual
        residual = q_hidden_states

        q_hidden_states = self.post_attention_layernorm(q_hidden_states)
        q_hidden_states = self.mlp(q_hidden_states)
        q_hidden_states = self.mlp_dropout(q_hidden_states)

        q_hidden_states = q_hidden_states + residual

        outputs = (q_hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs

class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`MplugOwlVisionEncoderLayer`].

    Args:
        config (`MplugOwlVisionConfig`):
            The corresponding vision configuration for the `MplugOwlEncoder`.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
    
class TransformerDecoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`MplugOwlVisionEncoderLayer`].

    Args:
        config (`MplugOwlVisionConfig`):
            The corresponding vision configuration for the `MplugOwlEncoder`.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([TransformerDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        input_q_embeds,
        input_kv_embeds,
        q_attention_mask: Optional[torch.Tensor] = None,
        kv_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # important here
        hidden_states = input_q_embeds
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (input_kv_embeds,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    input_kv_embeds,
                    q_attention_mask,
                    kv_attention_mask,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    input_kv_embeds,
                    q_attention_mask,
                    kv_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
    
class TransformerTextDecoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`MplugOwlVisionEncoderLayer`].

    Args:
        config (`MplugOwlVisionConfig`):
            The corresponding vision configuration for the `MplugOwlEncoder`.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([TransformerDecoderLayer(config) for _ in range(config.num_decoder_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        input_q_embeds,
        input_kv_embeds,
        q_attention_mask: Optional[torch.Tensor] = None,
        kv_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # important here
        hidden_states = input_q_embeds
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (input_kv_embeds,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    input_kv_embeds,
                    q_attention_mask,
                    kv_attention_mask,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    input_kv_embeds,
                    q_attention_mask,
                    kv_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )