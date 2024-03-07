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
    

class DistributedAlignmentSearch1d(nn.Module):
    """
    1d version for testing
    """    

    def __init__(
        self,
        d_model,
    ):
        super().__init__()
        self.vector = nn.Parameter(torch.randn(1, d_model))
        
    def forward(self, o_orig, o_new):
        vector = self.vector / self.vector.norm()
        orig_projection = (o_orig @ vector.T) @ vector
        new_projection = (o_new @ vector.T) @ vector
        return o_orig - orig_projection + new_projection
            
        
class DistributedAlignmentSearch(nn.Module):

    def __init__(
        self,
        d_model,
        d_subspace
    ):
        super().__init__()
        self.subspace = nn.Parameter(torch.randn(d_subspace, d_model))
        self.gram_schmidt_orthogonalization()

    def forward(self, o_orig, o_new):
        orig_projection = (o_orig @ self.subspace.T) @ self.subspace
        new_projection = (o_new @ self.subspace.T) @ self.subspace
        return o_orig - orig_projection + new_projection
        
    def gram_schmidt_orthogonalization(self):
        with torch.no_grad():
            for i in range(self.subspace.size(0)):
                for j in range(i):
                    self.subspace.data[i] -= self.subspace.data[i].dot(self.subspace.data[j]) * self.subspace.data[j]
                self.subspace.data[i] = F.normalize(self.subspace.data[i], dim=0)


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
    #logprobs1 = F.log_softmax(logits1, dim=-1)
    #logprobs2 = F.log_softmax(logits2, dim=-1)
    logit_diff_1 = logits1[:, indices[0]]  - logits1[:, indices[1]]
    logit_diff_2 = logits2[:, indices[0]]  - logits2[:, indices[1]]
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
