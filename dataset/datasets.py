import torch
import numpy as np
import os
import json
import pickle
import glob
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Dict, Optional, Sequence, List, Any, Union
from utils.registry import DATASET_REGISTRY,COLLATE_REGISTRY

from icecream import ic
from tqdm import tqdm

from dataset.common import load_video, load_image
from dataset.preprocess.process_drama import drama_annotation_process
from dataset.processor.ImageCaption_processor import ImageCaptionProcessor

def load_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]
    
@DATASET_REGISTRY.register()
class drama_dataset(Dataset):
    """
    requirements:
    1. data_arg
    2. split dataset
    3. tokenizer(Optional)

    Note:
    1. If the data not used in model input, the column will be removed unless
    """
    def __init__(self, data_args, split:str) -> None:
        super().__init__()
        self.split = split
        self.processor = ImageCaptionProcessor()

        self.dataset = []
        if isinstance(data_args.dataset_input_files, str):
            input_files = [data_args.dataset_input_files]
        for input_file in input_files:
            self.dataset += load_jsonl(input_file+"/"+split+".jsonl")

        # self.dataset = self.dataset[:int(len(self.dataset)*0.05)]
        
    def __getitem__(self, index) -> Any:
        data = self.dataset[index]
        image = data['img']
        if isinstance(image, str):
            image = load_image(image)
        caption = data['caption']
        pixel_value, caption = self.processor(image, caption)
        bbox = torch.FloatTensor(data['bbox'])

        return {
            'pixel_values': pixel_value,
            'labels':{
                'caption': caption,
                'bbox': bbox
            }
        }

    def __len__(self) -> int:
        return len(self.dataset)
    
@COLLATE_REGISTRY.register()
class drama_dataset_collate_fn(object): # <name>_dataset_collate_fn
    """
    """
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor | Any]:
        pixel_value = torch.cat([_['pixel_values'].unsqueeze(0) for _ in instances],dim=0)
        bbox = torch.cat([_['labels']['bbox'].unsqueeze(0) for _ in instances],dim=0)

        
        tokenized = self.tokenizer(
            [_['labels']['caption'] for _ in instances],
            # truncation=True,
            padding='longest',
            # max_length=self.tokenizer,
            return_tensors='pt'
        )

        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        return {
            'pixel_values': pixel_value,
            'attention_mask': attention_mask,
            'labels':{
                'caption': input_ids,
                'bbox': bbox
            }
        }