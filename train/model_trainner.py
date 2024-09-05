import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union, Any, Tuple, Mapping
from transformers import Trainer
from icecream import ic

class custom_trainer(Trainer):
    """
    Exisiting Trainer is enough for handling normal tasks.
    
    This Module is prepared for futuer expansion (e.g. Custom Sampler, Overrided Evaluation Process)

    All trainer should inherit from this class in case future expansion.
    """
    def _get_train_sampler(self):
        """
        Note: If val_sampler is needed, set val_sampler
        """
        return super()._get_train_sampler()
    
    def create_optimizer(self):
        return super().create_optimizer()