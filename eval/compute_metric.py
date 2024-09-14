import numpy as np
import json
from typing import Optional
from transformers.tokenization_utils import PreTrainedTokenizer
from .labels.eval import text_only_language_eval
from .object_detection.eval import get_iou

from icecream import ic

def merge_dict(dict_a, dict_b):
    output = {}
    for key in dict_a.keys():
        output[key] = dict_a[key]
    for key in dict_b.keys():
        output[key] = dict_b[key]
    return output

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
    
class multireference_text_with_obj_metric:
    def __init__(self,tokenizer: Optional[PreTrainedTokenizer]) -> None:
        self.tokenizer = tokenizer

    def __call__(self,eval_input):
        """
        set to be called as a function
        """
        logits, labels = eval_input

        gt_captions = labels['caption']
        gt_bbox = labels['bbox']
        pred = np.argmax(logits['caption'],axis=2) # (B,T,V)
        pred = self.tokenizer.batch_decode(pred,skip_special_tokens=True)
        gt_captions = np.where(gt_captions != -100,gt_captions,self.tokenizer.pad_token_id)
        gt_captions = self.tokenizer.batch_decode(gt_captions,skip_special_tokens=True)
        output = text_only_language_eval(pred,gt_captions)

        output['Mean-IOU'] = float(get_iou(logits['bbox'],gt_bbox))

        json.dump(pred,open("../checkpoints/outputs/eval.json",mode='w'))

        return output