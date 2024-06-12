from typing import Dict, List
from torch.utils.data import Dataset
from transformers import Trainer

class custom_trainer(Trainer):
    """
    Exisiting Trainer is enough for handling normal tasks.
    
    This Module is prepared for futuer expansion (e.g. Custom Sampler, Overrided Evaluation Process)

    All trainer should inherit from this class in case future expansion.
    """
    def evaluate(self, eval_dataset: Dataset | None = None, ignore_keys: List[str] | None = None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)