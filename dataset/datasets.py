import torch
import glob
import numpy as np
import os
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Dict, Optional, Sequence, List, Any, Union
from fvcore.common.registry import Registry

from icecream import ic
from tqdm import tqdm
from .common import load_video
from .process_msrvtt import msrvtt_annotation_process
from .process_bddx import bddx_annotation_process

DATASET_REGISTRY = Registry("DATASET")
COLLATE_REGISTRY = Registry("COLLATE")

@DATASET_REGISTRY.register()
class bddx_dataset(Dataset): # <name>_dataset
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
        self.video_2d_path = data_args.video_2d_path
        self.video_3d_path = data_args.video_3d_path
        self.video_object_path = data_args.video_object_path
        self.caption_file_path = data_args.caption_file_path

        self.annotations, self.name_list = bddx_annotation_process(self.caption_file_path, self.split)

        self.item_list = []
        for annotation in self.annotations:
            video_name = annotation['video_name'][:-4]
            # sentence = annotation['sentence']
            action = annotation['action']
            reason = annotation['reason']
            self.item_list.append((video_name, action, reason))
        
    def __getitem__(self, index) -> Any:
        video_name, action, reason = self.item_list[index]
        # video_tensor = load_video(path,self.video_length,False,224,4,False)
        input_2d = torch.from_numpy(np.load(os.path.join(self.video_2d_path,video_name+".npy"))).float()
        input_3d = torch.from_numpy(np.load(os.path.join(self.video_3d_path,video_name+".npy"))).float()
        input_object = torch.from_numpy(np.load(os.path.join(self.video_object_path,video_name+".npy"))).float()

        return {
            'input_2d': input_2d,
            'input_3d': input_3d,
            'input_object': input_object,
            'labels': {
                'action': action,
                'reason': reason,
            }
        }

    def __len__(self) -> int:
        return len(self.item_list)
    
@COLLATE_REGISTRY.register()
class bddx_dataset_collate_fn(object): # <name>_dataset_collate_fn
    """
    """
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor | Any]:

        input_2d = torch.cat([_['input_2d'].unsqueeze(0) for _ in instances],dim=0)
        input_3d = torch.cat([_['input_3d'].unsqueeze(0) for _ in instances],dim=0)
        input_object = torch.cat([_['input_object'].unsqueeze(0) for _ in instances],dim=0)

        action = self.tokenizer(
            [_['labels']['action'] for _ in instances],
            truncation=True,
            padding='longest',
            # max_length=self.each_cap_len,
            return_tensors='pt'
        )['input_ids']

        reason = self.tokenizer(
            [_['labels']['reason'] for _ in instances],
            truncation=True,
            padding='longest',
            # max_length=self.each_cap_len,
            return_tensors='pt'
        )['input_ids']
        
        return {
            'input_2d': input_2d,
            'input_3d': input_3d,
            'input_object': input_object,
            'labels':{
                'action': action,
                'reason': reason,
            }
        }

    