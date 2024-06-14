import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Dict, Optional, Sequence, List, Any, Union
from fvcore.common.registry import Registry

from icecream import ic

DATASET_REGISTRY = Registry("DATASET")
COLLATE_REGISTRY = Registry("COLLATE")

@DATASET_REGISTRY.register()
class custom_dataset(Dataset):
    """
    requirements:
    1. data_arg
    2. split dataset
    3. tokenizer(Optional)
    """
    def __init__(self,data_arg,split:str ,tokenizer: Optional[PreTrainedTokenizer] = None) -> None:
        super().__init__()

    def __getitem__(self, index) -> Dict[str : Union[torch.Tensor, Any]]:
        pass

    def __len__(self) -> int:
        pass

@COLLATE_REGISTRY.register()
class custom_dataset_collate_fn(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        pass
