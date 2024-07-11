import torch
import json
import torch.utils.data
import glob
import numpy as np
from icecream import ic
from typing import Tuple

def annotation_process(caption_path:str, split:str):
    """
    return a list of tuple -> (caption,name) / train_video_id_list or val_video_id_list
    """
    with open(caption_path,mode='r') as p:
        json_file = json.load(p)
        videos = json_file['videos']
        sentences = json_file['sentences']
        
    train_video_list = []
    val_video_list = []
    name2cap = {}
    for video_info in videos:
        if video_info['split'] == 'train':
            train_video_list.append(video_info['video_id'])
        elif video_info['split'] == 'validate':
            val_video_list.append(video_info['video_id'])

    for sentence in sentences:
        caption = sentence['caption']
        name = sentence['video_id']
        if name not in name2cap.keys():
            name2cap[name] = [caption]
        else:
            name2cap[name].append(caption)

    return name2cap, train_video_list if split == "train" else val_video_list