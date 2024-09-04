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
        self.caption_length = data_args.caption_seq_len
        self.video_length = data_args.video_seq_len
        self.video_folder_path = data_args.video_folder_path
        self.caption_file_path = data_args.caption_file_path

        self.annotations, self.name_list = bddx_annotation_process(self.caption_file_path, self.split)

        self.item_list = []
        for annotation in self.annotations:
            video_name = annotation['video_name']
            video_path = os.path.join(self.video_folder_path,video_name)
            sentence = annotation['sentence']
            self.item_list.append((video_path,sentence))
        

    def __getitem__(self, index) -> Any:
        path, sentence = self.item_list[index]
        video_tensor = load_video(path,self.video_length,False,224,4,False)
        
        return {
            'video_tensor': video_tensor,
            'labels': sentence
        }

    def __len__(self) -> int:
        return len(self.annotations)
    
@COLLATE_REGISTRY.register()
class bddx_dataset_collate_fn(object): # <name>_dataset_collate_fn
    """
    """
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor | Any]:
        video_tensor = torch.cat([_['video_tensor'] for _ in instances],dim=0)
        caption = [_['caption'] for _ in instances]

        return {
            'video_tensor': video_tensor,
            'labels': caption
        }

    