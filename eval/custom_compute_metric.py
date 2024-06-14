import numpy as np
from typing import Optional
from transformers.tokenization_utils import PreTrainedTokenizer
from ..eval.eval_pack.eval import text_only_language_eval

from icecream import ic

def custom_compute_metric(eval_input,tokenizer: Optional[PreTrainedTokenizer] = None):
    #pred -> batch of single sentences -> ndarry (B(stacked),T,V)
    #labels -> batch of a group of reference sentences -> List[List[str],...]
    assert tokenizer is not None
    logits, labels = eval_input
    pred = np.argmax(logits,axis=2) # (B,T,V)
    pred = tokenizer.batch_decode(pred)
    labels = tokenizer.batch_decode(labels)

    return text_only_language_eval(pred,labels)