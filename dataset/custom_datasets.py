import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence, List
from fvcore.common.registry import Registry

from icecream import ic

DATASET_REGISTRY = Registry("DATASET")
COLLATE_REGISTRY = Registry("COLLATE")

@DATASET_REGISTRY.register()
class custom_dataset(Dataset):
    """
    requirements:
    1. data_arg
    """
    def __init__(self,data_arg,split) -> None:
        super().__init__()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

@COLLATE_REGISTRY.register()
class custom_dataset_collate_fn(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        pass
