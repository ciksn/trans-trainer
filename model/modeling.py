from typing import Tuple,Dict
import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput,CausalLMOutput
from model.modeling_transformer import DownSampleMLP
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
    def __init__(self, config: PretrainedConfig, tokenizer: PreTrainedTokenizer | None = None, *inputs, **kwargs):
        super(custom_model,self).__init__(config, *inputs, **kwargs)
        self.config = config
        self.tokenizer = tokenizer
        self.loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX,label_smoothing=config.label_smoothing)
        self.SceneLevel = SceneLevel(config)
        self.ObjectLevel = ObjectLevel(config)
        self.MotionLevel = MotionLevel(config)

        self.module_downsample = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.input_downsample = nn.Linear(config.dim_2d + config.dim_3d + config.dim_object, config.hidden_size)
        self.mixed_downsample = DownSampleMLP(2 * config.hidden_size, config.hidden_size,config)

        self.text_generation = TextGeneration(config)

    def get_tokenizer(self,) -> PreTrainedTokenizer:
        return self.tokenizer
        
    def _prepare_model_inputs(self, inputs: torch.Tensor | None = None, bos_token_id: int | None = None, model_kwargs: torch.Dict[str, torch.Tensor] | None = None) -> Tuple[Tensor, str | None, Dict[str, Tensor]]:
        return super()._prepare_model_inputs(inputs, bos_token_id, model_kwargs)

    def forward(self,input_2d, input_3d, input_object, labels):
        """
        The inputs need to be flattened here since (**input) when called
        The output of the main model should be "ModelOutput"
        or dict contains key "loss" / 1st element be loss

        Output is a Dict
        """
        scene_output = self.SceneLevel(input_2d, input_3d)
        object_output = self.ObjectLevel(input_object, scene_output)
        motion_output = self.MotionLevel(scene_output,object_output)

        mixed_output = self.module_downsample(torch.cat((scene_output,object_output,motion_output[:,:32,:]),dim=-1))
        mixed_feat = self.input_downsample(torch.cat((input_2d,input_3d,input_object),dim=-1))
        actual_visual_input = self.mixed_downsample(torch.cat((mixed_output,mixed_feat),dim=-1))

        #TODO below
        input_ids = self.tokenizer(
            labels,
            truncation=True,
            padding='max_length',
            max_length=self.config.caption_seq_len,
            return_tensors='pt')['input_ids'].to(actual_visual_input.device)
        
        logits = self.text_generation(actual_visual_input,input_ids[...,:-1],None)

        #TODO Trick: consider label smoothing here
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            loss = self.loss(shift_logits, shift_labels)

        #when computing loss, remember to distinct single batch when training or batch of list when eval

        return CausalLMOutput(
            loss = loss,
            logits = logits
        )
    
    @torch.no_grad()
    def generate(self,input_2d, input_3d, input_object):
        pass

if __name__ == "__main__":
    device = "cuda:0"
    # Test code here -> to be REMOVED
    tokenizer = AutoTokenizer.from_pretrained("/home/zeyu/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594")
    custom_model_config_b = custom_model_config.from_pretrained("/home/zeyu/work/deep_learning/functional_files/trans_trainer/test/config.json")
    model = custom_model(custom_model_config_b, tokenizer).to(device)
    input_2d = torch.ones((1,32,768)).float().to(device)
    input_3d = torch.ones((1,32,1024)).float().to(device)
    input_object = torch.ones((1,32,2048)).float().to(device)
    labels = ["a man is eating"]

    output = model.forward(input_2d,input_3d,input_object,labels)