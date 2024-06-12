# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
from transformers.models.clip.image_processing_clip import CLIPImageProcessor

from torch.utils.data import Dataset
from ..train.custom_model_trainner import custom_trainer
from ..constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from ..dataset.custom_datasets import DATASET_REGISTRY,COLLATE_REGISTRY

from PIL import Image
from icecream import ic

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    """
    Customizable for train/val/test model Arguments
    """
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)

@dataclass
class DataArguments:
    """
    Customizable for train/val/test data Arguments
    """

    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    dataset_name: str = field(default=None,metadata={"help":"The dataset used for training"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    
    tune_visual_abstractor: bool = field(default=True)
    freeze_vision_model: bool = field(default=True)

    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    visual_abstractor_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['vision_model', 'visual_abstractor']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def build_data_module(data_args) -> Dict:
    """
    Get dataset and collator function for training
    """
    train_dataset = DATASET_REGISTRY.get(data_args.dataset_name)(data_args,"train")
    eval_dataset = DATASET_REGISTRY.get(data_args.dataset_name)(data_args,"eval")
    test_dataset = DATASET_REGISTRY.get(data_args.dataset_name)(data_args,"test")
    collator = COLLATE_REGISTRY.get(data_args.dataset_name+"_collate_fn")()
    return dict(
        train_dataset = train_dataset, # TODO if there needs quote
        eval_dataset = eval_dataset,
        test_dataset = test_dataset,
        data_collator = collator
    )


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)) # Auto assgin args based on class field 
    model_args, data_args, training_args = parser.parse_args_into_dataclasses() 

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    model = MPLUGOwl2LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args
    )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    
    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    
    if not training_args.freeze_vision_model and training_args.bits in [4, 8]:
        model.get_model().vision_model.to(dtype=compute_dtype, device=training_args.device)
    else:
        vision_tower = model.get_model().vision_model
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    
    if training_args.tune_visual_abstractor and training_args.bits in [4, 8]:
        model.get_model().visual_abstractor.to(dtype=compute_dtype, device=training_args.device)
    else:
        visual_abstractor = model.get_model().visual_abstractor
        visual_abstractor.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    
    data_args.image_processor = CLIPImageProcessor.from_pretrained(model_args.model_name_or_path)
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
    model.config.tune_visual_abstractor = model_args.tune_visual_abstractor = training_args.tune_visual_abstractor
    ic(training_args.tune_visual_abstractor)
    model.requires_grad_(True)
    if training_args.tune_visual_abstractor:
        # model.requires_grad_(False)
        for p in model.get_model().visual_abstractor.parameters():
            p.requires_grad = True
            
    model.config.freeze_vision_model = training_args.freeze_vision_model
    ic(training_args.freeze_vision_model)
    if training_args.freeze_vision_model:
        for p in model.get_model().vision_model.parameters():
            p.requires_grad = False
            
    model.config.visual_abstractor_lr = training_args.visual_abstractor_lr


    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = build_data_module(data_args)

    trainer = custom_trainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    #     trainer.train()
    
    # TODO I dont like auto resume << REMOVE IT AND UNCOMMENT THE ABOVE CODE
    trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()