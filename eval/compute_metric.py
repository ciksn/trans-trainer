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
        all_output = {}
        logits, labels = eval_input
        pred_action = np.argmax(logits['action'],axis=2) # (B,T,V)
        pred_action = self.tokenizer.batch_decode(pred_action,skip_special_tokens=True)

        labels['action'] = np.where(labels['action'] != -100,labels['action'],self.tokenizer.pad_token_id)
        labels_action = self.tokenizer.batch_decode(labels['action'],skip_special_tokens=True)
        output_action = text_only_language_eval(pred_action,labels_action)

        pred_reason = np.argmax(logits['reason'],axis=2) # (B,T,V)
        pred_reason = self.tokenizer.batch_decode(pred_reason,skip_special_tokens=True)
        labels['reason'] = np.where(labels['reason'] != -100,labels['reason'],self.tokenizer.pad_token_id)
        labels_reason = self.tokenizer.batch_decode(labels['reason'],skip_special_tokens=True)
        output_reason = text_only_language_eval(pred_reason,labels_reason)

        for key in output_action.keys():
            all_output["action/"+key] = output_action[key]
        for key in output_reason.keys():
            all_output["reason/"+key] = output_reason[key]

        action_json = []
        for i in range(len(pred_action)):
            action_json.append({
                'pred': pred_action[i],
                'gt': labels_action[i]
            })
        
        reason_json = []
        for i in range(len(pred_reason)):
            reason_json.append({
                'pred': pred_reason[i],
                'gt': labels_reason[i]
            })
            
        # json.dump(action_json,open("../checkpoints/outputs/action.json",mode='w+'))
        # json.dump(reason_json,open("../checkpoints/outputs/reason.json",mode='w+'))
        return all_output