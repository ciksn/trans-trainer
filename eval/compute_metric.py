import numpy as np
from typing import Optional
from transformers.tokenization_utils import PreTrainedTokenizer
from .labels.eval import text_only_language_eval

from icecream import ic

class multireference_text_metric:
    def __init__(self,tokenizer: Optional[PreTrainedTokenizer]) -> None:
        assert tokenizer is not None
        self.tokenizer = tokenizer

    def __call__(self,eval_input):
        """
        set to be called as a function
        """
        logits, labels = eval_input
        pred = np.argmax(logits,axis=2) # (B,T,V)
        pred = self.tokenizer.batch_decode(pred)
        labels = self.tokenizer.batch_decode(labels)

        return text_only_language_eval(pred,labels)