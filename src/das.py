import numpy as np
from functools import partial
from tqdm import tqdm
import random
import json

import torch
from torch import nn
import torch.nn.functional as F


class DistributedAlignmentSearch(nn.Module):
    """
    Trainable subspace that we can optimize over
    """    

    def __init__(
        self,
        d_model,
        d_subspace
    ):
        super().__init__()
        self.subspace = nn.utils.parametrizations.orthogonal(
            nn.Linear(d_model, d_subspace, bias=False)
        )
        
    def forward(self, o_orig, o_new):
        orig_projection = (o_orig @ self.subspace.weight.T) @ self.subspace.weight
        new_projection = (o_new @ self.subspace.weight.T) @ self.subspace.weight
        return o_orig - orig_projection + new_projection
    

def sample_contrast_triplets(task_obj, dataset_path, num_examples):
    # Sample num_examples * 2 dataset points
    data = []
    with open(dataset_path, 'r') as f:
        lines = f.readlines()
        sampling = random.sample(lines, num_examples * 2)
        for line in sampling:
            data.append(json.loads(line))
    data = [d["prompt"] for d in data]
    # Generate contrast examples
    examples = {
        "clean": [],
        "persona_diff": [],
        "seq_diff": [],
    }
    for i in range(num_examples):
        # Choose two different personas
        sampled_p = random.sample(sorted(task_obj.personas), 2)
        examples["clean"].append(
            task_obj.personas[sampled_p[0]] + task_obj.prompt.format(sequence=data[i])
        )
        examples["persona_diff"].append(
            task_obj.personas[sampled_p[1]] + task_obj.prompt.format(sequence=data[i])
        )
        examples["seq_diff"].append(
            task_obj.personas[sampled_p[0]] + task_obj.prompt.format(sequence=data[i+num_examples])
        )
    return examples


def patching_metric(logits1, logits2, indices=(4986, 29907)):
    logprobs1 = F.log_softmax(logits1, dim=-1)
    logprobs2 = F.log_softmax(logits2, dim=-1)
    idx1_diff = torch.pow(logprobs1[:, indices[0]] - logprobs2[:, indices[0]], 2).mean()
    idx2_diff = torch.pow(logprobs1[:, indices[1]] - logprobs2[:, indices[1]], 2).mean()
    return idx1_diff + idx2_diff
    
