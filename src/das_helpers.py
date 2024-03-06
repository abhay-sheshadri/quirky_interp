import numpy as np
import random
import json
from functools import partial
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from .patching_helpers import tokenize_examples


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
    logit_diff_1 = logprobs1[:, indices[0]]  - logprobs1[:, indices[1]]
    logit_diff_2 = logprobs2[:, indices[0]]  - logprobs2[:, indices[1]]
    return torch.pow(logit_diff_1 - logit_diff_2, 2).mean()
    

class ConstrastTriplesDataset(Dataset):
    def __init__(self, model, task, dataset_path, n_examples=750):
        super(Dataset, self).__init__()
        self.samples = sample_contrast_triplets(task, dataset_path, n_examples)
        self.token_samples = {}
        for key in self.samples:
            tokens, indices = tokenize_examples(self.samples[key], model)
            self.token_samples[key+"_tokens"] = tokens
            self.token_samples[key+"_indices"] = indices

    def __len__(self):
        # Returns the length of the dataset
        return len(next(iter(self.token_samples.values())))

    def __getitem__(self, index):
        # Returns data and label at the given index as a dictionary
        return {k: torch.Tensor(v[index]) for k, v in self.token_samples.items()}
    
    
def patching_hook(acts, hook, acts_idx, new_acts, new_acts_idx, das):
    batch_size = acts.shape[0]
    o_orig = acts[torch.arange(batch_size), acts_idx]
    o_new = new_acts[torch.arange(batch_size), new_acts_idx]
    acts[torch.arange(batch_size), acts_idx] = das(o_orig, o_new)
    return acts