import torch
import numpy as np
import torch.nn as nn
import argparse
import json
from tqdm import tqdm
from typing import Dict
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer, PreTrainedTokenizer
from dataset.datasets import DATASET_REGISTRY,COLLATE_REGISTRY
from model.configuration_model import MAINconfig
from model.modeling import MAIN
from eval.compute_metric import text_only_language_eval

from icecream import ic

def build_test_data_module(data_args,tokenzier:PreTrainedTokenizer = None) -> Dict:
    """
    Get dataset and collator function for training
    """
    test_dataset = DATASET_REGISTRY.get(data_args.dataset_name)(data_args, "test")
    collator = COLLATE_REGISTRY.get(data_args.dataset_name+"_collate_fn")(tokenzier)
    return DataLoader(
        dataset=test_dataset, 
        batch_size=data_args.batch_size,
        num_workers=data_args.num_workers,
        collate_fn=collator,
        pin_memory=data_args.pin_memory
        )

# TODO 
# Multi-GPU Support 
def main(config):
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint,add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    model_config = MAINconfig.from_pretrained(config.checkpoint)
    model = MAIN.from_pretrained(pretrained_model_name_or_path=config.checkpoint,config=model_config)
    model.to(config.device)

    test_dataloader = build_test_data_module(config,tokenizer)

    pred_list = []
    gt_list = []
    for batched in tqdm(test_dataloader):
        pixel_values = batched['pixel_values'].to(config.device)
        attention_mask = batched['attention_mask'].to(config.device)
        labels = batched['labels']

        caption = labels['caption']
        gt_list += tokenizer.batch_decode(caption,skip_special_tokens=True)

        args = {
            'max_new_tokens': 30,
            'num_beams': 5,
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.9,
            'do_sample': True,
        }

        pred_outputs = model.generate(
            pixel_values, 
            attention_mask,
            **args
        )

        pred_list += tokenizer.batch_decode(pred_outputs,skip_special_tokens=True)

    _ = text_only_language_eval(pred_list,gt_list)

    json_list = []
    for i in range(len(pred_list)):
        json_list.append({
            'pred': pred_list[i],
            'gt': gt_list[i]
        })

    json.dump(json_list, open("../checkpoints/outputs/test.json",mode='w'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/home/zeyu/work/deep_learning/functional_files/trans_trainer/checkpoints/checkpoint')
    parser.add_argument('--dataset_name', type=str, default='drama_dataset')
    parser.add_argument('--dataset_input_files', type=str, default='/home/zeyu/mnt/drive0/dataset/driving/drama/processed')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--device', type=str, default="cuda:0")
    config = parser.parse_args()
    main(config)