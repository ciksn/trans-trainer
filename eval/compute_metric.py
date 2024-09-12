import numpy as np
import json
from typing import Optional
from transformers.tokenization_utils import PreTrainedTokenizer
from .labels.eval import text_only_language_eval

from icecream import ic

class multireference_text_metric:
    def __init__(self,tokenizer: Optional[PreTrainedTokenizer]) -> None:
        self.tokenizer = tokenizer

    def __call__(self,eval_input):
        """
        set to be called as a function
        """
        logits, labels = eval_input
        labels = labels['caption']
        pred = np.argmax(logits,axis=2) # (B,T,V)
        pred = self.tokenizer.batch_decode(pred,skip_special_tokens=True)
        labels = np.where(labels != -100,labels,self.tokenizer.pad_token_id)
        labels = self.tokenizer.batch_decode(labels,skip_special_tokens=True)
        output = text_only_language_eval(pred,labels)

        json.dump(pred,open("../checkpoints/outputs/eval.json",mode='w'))

        return output