# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import os
import json

try:
    from tensorboardX import SummaryWriter
except ImportError:
    print("Cannot import tensorboard. Will log to txt files only.")
    SummaryWriter = None

from utils.dist import is_primary



def log_evaluation_results(message, logout):
    logout(message)

def save_evaluation_results(checkpoint_dir, corpus, candidates, score_per_caption):
    with open(os.path.join(checkpoint_dir, "corpus_val.json"), "w") as f: 
        json.dump(corpus, f, indent=4)
    
    with open(os.path.join(checkpoint_dir, "pred_val.json"), "w") as f:
        json.dump(candidates, f, indent=4)
    
    with open(os.path.join(checkpoint_dir, "pred_gt_val.json"), "w") as f:
        pred_gt_val = {
            scene_object_id_key: {
                'pred': candidates[scene_object_id_key],
                'gt': corpus[scene_object_id_key],
                'score': {
                    'bleu-1': score_per_caption['bleu-1'][scene_object_id],
                    'bleu-2': score_per_caption['bleu-2'][scene_object_id],
                    'bleu-3': score_per_caption['bleu-3'][scene_object_id],
                    'bleu-4': score_per_caption['bleu-4'][scene_object_id],
                    'CiDEr': score_per_caption['cider'][scene_object_id],
                    'rouge': score_per_caption['rouge'][scene_object_id],
                    'meteor': score_per_caption['meteor'][scene_object_id]
                }
            } for scene_object_id, scene_object_id_key in enumerate(candidates)
        }
        json.dump(pred_gt_val, f, indent=4)



class Logger(object):
    def __init__(self, log_dir=None) -> None:
        self.log_dir = log_dir
        if SummaryWriter is not None and is_primary():
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None

    def log_scalars(self, scalar_dict, step, prefix=None):
        if self.writer is None:
            return
        for k in scalar_dict:
            v = scalar_dict[k]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().item()
            if prefix is not None:
                k = prefix + k
            self.writer.add_scalar(k, v, step)
