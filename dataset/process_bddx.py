import torch
import numpy as np
import json
from icecream import ic
from typing import Tuple,List,Dict

def bddx_annotation_process(path: str, split: str) -> Tuple[List[Dict],List[str]]:
    with open(path,mode='r') as f:
        json_file = json.load(path)
    metadata = json_file['metadata']
    target_list = json_file[split]

    annotations = []
    for raw_video_info in metadata:
        if raw_video_info['raw_video'] in target_list:
            for video_part in raw_video_info['parts']:
                annotations.append({
                    'video_name': video_part['video_part'],
                    'action':video_part['action'],
                    'reason':video_part['reason'],
                    'sentence':video_part['sentence']
                })

    return annotations, target_list