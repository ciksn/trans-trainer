from os import PathLike
from typing import Tuple,Dict,Optional
import torch
import torch.nn as nn
from torch import Tensor
from transformers import (
    PreTrainedModel,
    LlamaForCausalLM,
    LlamaTokenizer,
    CLIPModel,
    PreTrainedTokenizerFast,
    PreTrainedTokenizer
)
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput,CausalLMOutput
from transformers.pipelines import AutoTokenizer,AutoConfig,AutoModel

from model.configuration_model import MAINconfig
from model.modeling_abstractor import MAINVisualAbstractorModel
from model.modeling_multitask import MAINMultiTaskModelForALL

from icecream import ic

class MAIN(PreTrainedModel):
    """
    Model definition here

    Input:
        Contains input data, label
    Output:
        Inherit from ModelOutput -> the first element must be loss

    """
    config_class = MAINconfig

    def __init__(
        self, 
        config: Optional[MAINconfig] = None, 
        *inputs, 
        **kwargs
        ):
        super(MAIN,self).__init__(config, *inputs, **kwargs)
        self.config = config


        self.visual_abstractor = MAINVisualAbstractorModel(config.visual_abstractor_config,config.language_model_config.hidden_size)
        self.visual_backbone = CLIPModel(config.visual_backbone_config).vision_model
        self.language_model = LlamaForCausalLM(config.language_model_config)

        #TODO 1
        self.multi_task = MAINMultiTaskModelForALL(config.multi_task_config)

        self.loss = nn.CrossEntropyLoss()

        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.visual_abstractor_config.hidden_size)
        )

        # self.sigma_1 = nn.Parameter(torch.rand(1)[0])
        # self.sigma_2 = nn.Parameter(torch.rand(1)[0])
        
    def _prepare_model_inputs(self, inputs: torch.Tensor | None = None, bos_token_id: int | None = None, model_kwargs: torch.Dict[str, torch.Tensor] | None = None) -> Tuple[Tensor, str | None, Dict[str, Tensor]]:
        return super()._prepare_model_inputs(inputs, bos_token_id, model_kwargs)

    def _prepare_inputs_with_pixel_for_training(
            self, 
            visual_hidden_states: torch.Tensor, 
            input_ids: torch.Tensor, 
            attention_mask:torch.Tensor
        ):

        # gt = input_ids.masked_fill(attention_mask == 0, -100)[:,1:]
        gt = input_ids[:,1:]
        actual_input_ids = input_ids[:,:-1]

        batch, T, _ = visual_hidden_states.size()
        batch, K = actual_input_ids.size()
        total_number = T + K

        # mask = torch.ones((total_number, total_number),device=visual_hidden_states.device)
        # casual_mask = torch.tril(torch.ones((K, K),device=visual_hidden_states.device))
        # zero_mask = torch.zeros((T,K),device=visual_hidden_states.device)
        # mask[T:,T:] = casual_mask
        # mask[:T,T:] = zero_mask
        # mask = mask.expand((batch,1,-1,-1))

        # mask = mask.masked_fill(mask == 0, float('-inf'))
        # mask = mask.masked_fill(mask == 1, 0)
        
        mask = torch.tril(torch.ones((total_number, total_number),device=visual_hidden_states.device))
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0)
        mask = mask.expand((batch, 1, -1, -1))

        word_embeddings = self.language_model.model.embed_tokens(actual_input_ids)
        input_embeds = torch.cat([visual_hidden_states, word_embeddings],dim=1)

        return (input_embeds, mask, gt)
    
    def _prepare_inputs_with_pixel_for_generation(
            self, 
            visual_hidden_states: torch.Tensor, 
            input_ids: torch.Tensor, 
        ):

        actual_input_ids = input_ids

        batch, T, _ = visual_hidden_states.size()
        batch, K = actual_input_ids.size()
        total_number = T + K

        # mask = torch.ones((total_number, total_number),device=visual_hidden_states.device)
        # casual_mask = torch.tril(torch.ones((K, K),device=visual_hidden_states.device))
        # zero_mask = torch.zeros((T,K),device=visual_hidden_states.device)
        # mask[T:,T:] = casual_mask
        # mask[:T,T:] = zero_mask
        # mask = mask.expand((batch,1,-1,-1))
        mask = torch.ones((batch, total_number),device=visual_hidden_states.device)

        word_embeddings = self.language_model.model.embed_tokens(actual_input_ids)
        input_embeds = torch.cat([visual_hidden_states, word_embeddings],dim=1)

        return (input_embeds, mask)
    
    def _get_loss(self, loss_caption, loss_obj):
        # final_loss = 1 / (2 * torch.square(self.sigma_1)) * loss_caption + \
        # 1 / (2 * torch.square(self.sigma_2)) * loss_obj + torch.log(self.sigma_1 * self.sigma_2)
        lam = 0.5
        final_loss = loss_caption * lam + loss_obj * (1 - lam)
        return final_loss

    def forward(self,pixel_values, attention_mask, labels):
        """
        The inputs need to be flattened here since (**input) when called
        The output of the main model should be "ModelOutput"
        or dict contains key "loss" / 1st element be loss

        Output is a Dict
        """
        if pixel_values is not None:
            query_token_embeds = self.query_tokens.expand((pixel_values.size(0),-1,-1))

            hidden_states = self.visual_backbone(pixel_values)[0]

            multi_task_output = self.multi_task(hidden_states,labels)
            loss_obj, logits_bbox = multi_task_output[0], multi_task_output[1]

            hidden_states = self.visual_abstractor(
                query_embeds = query_token_embeds,
                encoder_hidden_states = hidden_states,
            )[0]
        
            inputs_embeds, attention_mask, gt = self._prepare_inputs_with_pixel_for_training(hidden_states,labels['caption'],attention_mask)

        logits_caption = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )[0][:,hidden_states.size(1):]

        if labels is not None:
            logits_caption = logits_caption[..., :, :].contiguous()
            shift_gt = gt[..., :].contiguous()
            logits_caption_flat = logits_caption.view(-1, self.config.language_model_config.vocab_size)
            shift_gt_flat = shift_gt.view(-1)

            shift_gt_flat = shift_gt_flat.to(logits_caption.device)
            loss_caption = self.loss.forward(logits_caption_flat, shift_gt_flat)

        loss = self._get_loss(loss_caption,loss_obj)

        # ic(loss.item())
        #when computing loss, remember to distinct single batch when training or batch of list when eval

        return CausalLMOutput(
            loss = loss,
            logits = {
                'bbox': logits_bbox,
                'caption': logits_caption
            }
        )
    
    @torch.no_grad()
    def generate(self, pixel_values, attention_mask=None, **generate_kwargs):
        """
        生成语言输出，使用语言模型自带的 `generate` 方法。
        
        参数：
        - pixel_values: 视觉输入的像素值
        - attention_mask: 注意力掩码 (optional)
        - max_length: 最大生成长度
        - generate_kwargs: 传递给生成方法的其他参数（如 `num_beams`，`do_sample` 等）

        返回：
        - 生成的 token 序列
        """
        # 提取视觉特征
        query_token_embeds = self.query_tokens.expand((pixel_values.size(0), -1, -1))
        hidden_states = self.visual_backbone(pixel_values)[0]

        multi_task_output = self.multi_task(hidden_states,None)
        loss_obj, logits_bbox = multi_task_output[0], multi_task_output[1]

        hidden_states = self.visual_abstractor(
            query_embeds=query_token_embeds,
            encoder_hidden_states=hidden_states,
        )[0]

        # 初始化生成的 input_ids 序列
        input_ids = torch.full(
            (pixel_values.size(0), 1), self.config.bos_token_id, dtype=torch.long, device=pixel_values.device
        )

        # 准备语言模型的输入嵌入
        inputs_embeds, attention_mask = self._prepare_inputs_with_pixel_for_generation(hidden_states, input_ids)
        position_ids = torch.arange(0,inputs_embeds.size(1),dtype=torch.long,device=inputs_embeds.device).expand(inputs_embeds.size(0),-1)

        # 使用语言模型自带的 `generate` 方法
        generated_ids = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs  # 额外参数传递给生成函数，比如 num_beams, do_sample
        )

        return ModelOutput(
            generated_ids = generated_ids,
            logits_bbox = logits_bbox,
        )
    
    def freeze_language(self,):
        for param in self.language_model.parameters():
            param.requires_grad = False

    def unfreeze_language(self,):
        for param in self.language_model.parameters():
            param.requires_grad = True


AutoConfig.register("MAIN",MAINconfig)
AutoModel.register(MAINconfig,MAIN)

if __name__ == "__main__":
    pass