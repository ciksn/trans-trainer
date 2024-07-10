import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Dict, Optional, Sequence, List, Any, Union
from fvcore.common.registry import Registry

from icecream import ic
from preprocess import *

DATASET_REGISTRY = Registry("DATASET")
COLLATE_REGISTRY = Registry("COLLATE")

@DATASET_REGISTRY.register()
class msrvtt_dataset(Dataset):
    """
    requirements:
    1. data_arg
    2. split dataset
    3. tokenizer(Optional)
    """
    def __init__(self,data_args, split:str, tokenizer: Optional[PreTrainedTokenizer] = None) -> None:
        super().__init__()
        self.split = split
        self.caption_length = data_args.caption_seq_len
        self.video_length = data_args.video_seq_len
        self.video_folder_path = data_args.video_folder_path
        self.caption_file_path = data_args.caption_file_path
        
        name2cap, name_list = annotation_process(self.caption_file_path,split=split)
        self.name2path = {}
        videos_path_list = glob.glob(self.video_folder_path + '/*.npy')
        for video_path in videos_path_list:
            vid_name = video_path.split('/')[-1][:-4]
            if vid_name in name_list:
                self.name2path[vid_name] = video_path
        
        self.item_list = []
        if split == "train":
            for name in name2cap.keys():
                for caption in name2cap[name]:
                    self.item_list.append((name,
                                           tokenizer(caption,
                                                     return_tensors='pt')))
        else:
            for name in name2cap.keys():
                self.item_list.append(name,
                                      tokenizer(name2cap[name],
                                                return_tensors='pt'))

    def __getitem__(self, index) -> Dict[str : Union[torch.Tensor, Any]]:
        name, t_label = self.item_list[index]
        video_path = self.name2path[name]
        vid_feat = np.load(video_path)
        
        return {
            'vid_feat': vid_feat,
            'label': t_label
        }

    def __len__(self) -> int:
        return len(self.item_list)

@COLLATE_REGISTRY.register()
class msrvtt_collate_fn(object):
    """
    """
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        labels = [_['label'] for _ in instances]
        vid_feat = [_['vid_feat'] for _ in instances]

        labels = torch.cat([torch.tensor(item,dtype=torch.uint8).unsqueeze(0)for item in labels],dim=0)
        vid_feat = torch.cat([torch.tensor(item,dtype=torch.float32).unsqueeze(0)for item in vid_feat],dim=0)

        return {
            "vid_feat": vid_feat,
            "labels": labels
        }