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
from eval.compute_metric import text_only_language_eval
from model.modeling import custom_model
from model.configuration_model import custom_model_config

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
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    model_config = custom_model_config.from_pretrained(config.checkpoint)
    model = custom_model.from_pretrained(config.checkpoint,config=model_config)
    model.to(config.device)

    test_dataloader = build_test_data_module(config,tokenizer)

    pred_action_list = []
    pred_reason_list = []
    gt_action_list = []
    gt_reason_list = []
    for batch in tqdm(test_dataloader):
        input_2d = batch['input_2d'].to(config.device)
        input_3d = batch['input_3d'].to(config.device)
        input_object = batch['input_object'].to(config.device)

        label_dict = batch['labels']
        gt_action_list += tokenizer.batch_decode(label_dict['action'],skip_special_tokens=True)
        gt_reason_list += tokenizer.batch_decode(label_dict['reason'],skip_special_tokens=True)

        pred_outputs = model.generate(input_2d,input_3d,input_object,50,tokenizer.bos_token_id,tokenizer.eos_token_id)
        pred_action_list += tokenizer.batch_decode(pred_outputs['action'],skip_special_tokens=True)
        pred_reason_list += tokenizer.batch_decode(pred_outputs['reason'],skip_special_tokens=True)

    _ = text_only_language_eval(pred_action_list,gt_action_list)
    _ = text_only_language_eval(pred_reason_list,gt_reason_list)

    action_json = []
    for i in range(len(pred_action_list)):
        action_json.append({
            'pred': pred_action_list[i],
            'gt': gt_action_list[i]
        })

    reason_json = []
    for i in range(len(pred_reason_list)):
        reason_json.append({
            'pred': pred_reason_list[i],
            'gt': gt_reason_list[i]
        })

    json.dump(action_json, open("./checkpoints/outputs/action.json",mode='w'))
    json.dump(reason_json, open("./checkpoints/outputs/reason.json",mode='w'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/home/zeyu/work/deep_learning/functional_files/trans_trainer/checkpoints/checkpoint')
    parser.add_argument('--dataset_name', type=str, default='bddx_dataset')
    parser.add_argument('--video_2d_path', type=str, default='/home/zeyu/work/deep_learning/extracted_dataset/bddx/CLIP-ViT_L14')
    parser.add_argument('--video_3d_path', type=str, default='/home/zeyu/work/deep_learning/extracted_dataset/bddx/S3D')
    parser.add_argument('--video_object_path', type=str, default='/home/zeyu/work/deep_learning/extracted_dataset/bddx/Fasterrcnn')
    parser.add_argument('--caption_file_path', type=str, default='/home/zeyu/mnt/drive0/dataset/driving/BDD-X/BDD-X-Dataset/bddx.json')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--device', type=str, default="cuda:0")
    config = parser.parse_args()
    main(config)