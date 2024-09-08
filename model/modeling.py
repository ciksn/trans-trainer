from typing import Tuple,Dict
import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput,CausalLMOutput
from model.modeling_modules import SceneLevel, ObjectLevel, MotionLevel, TextGeneration
from model.configuration_model import custom_model_config
from constants import IGNORE_INDEX

from transformers.pipelines import AutoTokenizer
from icecream import ic

class custom_model(PreTrainedModel):
    """
    Model definition here

    Input:
        Contains input data, label
    Output:
        Inherit from ModelOutput -> the first element must be loss

    """
    config_class = custom_model_config

    def __init__(self, config: PretrainedConfig, tokenizer: PreTrainedTokenizer | None = None, *inputs, **kwargs):
        super(custom_model,self).__init__(config, *inputs, **kwargs)
        self.config = config
        self.tokenizer = tokenizer
        self.loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX,label_smoothing=config.label_smoothing)
        self.SceneLevel = SceneLevel(config)
        self.ObjectLevel = ObjectLevel(config)
        self.MotionLevel = MotionLevel(config)

        self.module_downsample = nn.Linear(3 * config.hidden_size, config.hidden_size,config)

        self.text_generation = TextGeneration(config)

    def get_tokenizer(self,) -> PreTrainedTokenizer:
        return self.tokenizer
        
    def _prepare_model_inputs(self, inputs: torch.Tensor | None = None, bos_token_id: int | None = None, model_kwargs: torch.Dict[str, torch.Tensor] | None = None) -> Tuple[Tensor, str | None, Dict[str, Tensor]]:
        return super()._prepare_model_inputs(inputs, bos_token_id, model_kwargs)

    def get_media_mask(self,T, K):
        total_number = T+K

        mask = torch.ones((total_number, total_number))
        casual_mask = torch.tril(torch.ones((K,K)))
        mask[:K,T:] = casual_mask

        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0)
        return mask.unsqueeze(0)

    def forward(self,input_2d, input_3d, input_object, labels):
        """
        The inputs need to be flattened here since (**input) when called
        The output of the main model should be "ModelOutput"
        or dict contains key "loss" / 1st element be loss

        Output is a Dict
        """
        scene_output = self.SceneLevel(input_2d, input_3d)
        # object_output = self.ObjectLevel(input_object, scene_output)
        # motion_output = self.MotionLevel(scene_output,object_output)
        # actual_visual_input = self.module_downsample(torch.cat((scene_output,object_output,motion_output),dim=-1))
        actual_visual_input = scene_output

        actual_action_ids = labels['action'][...,:-1]
        actual_reason_ids = labels['reason'][...,:-1]

        B,T,H = actual_visual_input.size()
        action_T = actual_action_ids.size(1)
        reason_T = actual_reason_ids.size(1)

        logits_action, logits_reason = self.text_generation(
            actual_visual_input,
            actual_action_ids,
            actual_reason_ids,
            self.get_media_mask(T,action_T).to(actual_visual_input.device),
            None,
            self.get_media_mask(T,reason_T).to(actual_visual_input.device),
            None)

        hypert = 0.2
        #TODO Trick: consider label smoothing here
        loss = None
        if labels is not None:
            shift_logits = logits_action[..., :, :].contiguous()
            shift_labels = labels['action'][..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            loss = self.loss(shift_logits, shift_labels) * hypert

            shift_logits = logits_reason[..., :, :].contiguous()
            shift_labels = labels['reason'][..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            loss += self.loss(shift_logits, shift_labels) * (1-hypert)

        # ic(loss.item())
        #when computing loss, remember to distinct single batch when training or batch of list when eval

        return CausalLMOutput(
            loss = loss,
            logits = {
                'action': logits_action,
                'reason': logits_reason,
            }
        )
    
    @torch.no_grad()
    def generate(
        self, 
        input_2d: torch.Tensor, 
        input_3d: torch.Tensor, 
        input_object: torch.Tensor, 
        max_length=50, 
        bos_token_id=None, 
        eos_token_id=None):
        """
        Autoregressive generation of action and reason captions.
        Args:
            input_3d: 3D scene input
            input_object: Object-level input
            max_length: Maximum length of the generated sequence
            bos_token_id: ID of the beginning-of-sequence token
            eos_token_id: ID of the end-of-sequence token
        Returns:
            Dict containing the generated captions (action and reason).
        """
        self.eval()
        # Ensure BOS token is provided
        if bos_token_id is None:
            bos_token_id = self.tokenizer.bos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        # Step 1: Get the visual inputs by passing them through Scene, Object, and Motion levels
        scene_output = self.SceneLevel(input_2d, input_3d)
        # object_output = self.ObjectLevel(input_object, scene_output)
        # motion_output = self.MotionLevel(scene_output,object_output)
        # actual_visual_input = self.module_downsample(torch.cat((scene_output,object_output,motion_output),dim=-1))
        actual_visual_input = scene_output


        B,T,H = actual_visual_input.size()
    
        # Initialize with BOS token for action and reason
        action_generated = torch.tensor([[bos_token_id]]).expand((B,1)).to(actual_visual_input.device)
        reason_generated = torch.tensor([[bos_token_id]]).expand((B,1)).to(actual_visual_input.device)

        # Step 2: Iteratively generate tokens
        for step in range(max_length):
            # Apply causal masks for autoregressive generation
            action_T = action_generated.size(1)
            reason_T = reason_generated.size(1)

            # Step 3: Pass the inputs through the text generation module
            logits_action, logits_reason = self.text_generation(
                actual_visual_input,
                action_generated,
                reason_generated,
                self.get_media_mask(T,action_T).to(actual_visual_input.device),
                None,
                self.get_media_mask(T,reason_T).to(actual_visual_input.device),
                None
            )

            # Get the predicted next token (use greedy decoding here)
            next_action_token = torch.argmax(logits_action[:, -1, :], dim=-1)
            next_reason_token = torch.argmax(logits_reason[:, -1, :], dim=-1)
            # Append predicted token to the generated sequence
            action_generated = torch.cat([action_generated, next_action_token.unsqueeze(-1)], dim=-1)
            reason_generated = torch.cat([reason_generated, next_reason_token.unsqueeze(-1)], dim=-1)

            # Step 4: Check for EOS token to terminate early
            if (next_action_token == eos_token_id).all() and (next_reason_token == eos_token_id).all():
                break

        return {
            "action": action_generated,
            "reason": reason_generated,
        }


if __name__ == "__main__":
    device = "cuda:0"
    # Test code here -> to be REMOVED
    tokenizer = AutoTokenizer.from_pretrained("/home/zeyu/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594")
    custom_model_config_b = custom_model_config.from_pretrained("/home/zeyu/work/deep_learning/functional_files/trans_trainer/test/config.json")
    model = custom_model(custom_model_config_b, tokenizer).to(device)
    input_2d = torch.ones((1,32,768)).float().to(device)
    input_3d = torch.ones((1,32,1024)).float().to(device)
    input_object = torch.ones((1,32,2048)).float().to(device)
    labels = ["a man is eating","a woman is eating"]

    output = model.forward(input_2d,input_3d,input_object,labels)