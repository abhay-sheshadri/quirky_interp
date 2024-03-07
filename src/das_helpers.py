import numpy as np
import random
import json
import functools
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from .patching_helpers import tokenize_examples

        
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

def run_das_experiment(
        model,
        train_dataloader,
        test_dataloader,
        pos_list,
        n_dim,
        learning_rate,
        invariant_seq,
        invariant_persona,
        n_epochs,
        acc_step_batch_size,
        acc_iters,
        verbose=False,
):
    from datetime import datetime
    exp_time = datetime.now().strftime("%b%d-%H%M-%S")

    folder = f"das-experiment_seq-{invariant_seq}_persona-{invariant_persona}_{exp_time}"
    results = {}

    for layer in tqdm(model.cfg.n_layers):
        results[layer] = {}
        for position in pos_list:
            results[layer][position] = {}

            if verbose:
                print(f"Running experiment for layer {layer} and position {position}")

            linear_rep, train_seq_loss, train_persona_loss, test_seq_loss, test_persona_loss = train_linear_rep(
                model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                n_dim=n_dim,
                learning_rate=learning_rate,
                layer=layer,
                pos=position,
                invariant_seq=invariant_seq,
                invariant_persona=invariant_persona,
                n_epochs=n_epochs,
                acc_step_batch_size=acc_step_batch_size,
                acc_iters=acc_iters,
                verbose=False,
            )
            torch.save(linear_rep.subspace, f"{folder}/linear_rep_{layer}_{position}.pt")

            results[layer][position]["train_seq_loss"] = train_seq_loss
            results[layer][position]["train_persona_loss"] = train_persona_loss
            results[layer][position]["test_seq_loss"] = test_seq_loss
            results[layer][position]["test_persona_loss"] = test_persona_loss

            torch.save(results, f"{folder}/results.pt")


def train_linear_rep(
    model,
    train_dataloader,
    test_dataloader,
    n_dim,
    learning_rate,
    layer,
    pos,
    invariant_seq,
    invariant_persona,
    n_epochs,
    acc_step_batch_size,
    acc_iters,
    verbose=False,
):

    # Freeze model params
    for param in model.parameters():
        param.requires_grad_(False)
    names_filter = [f"blocks.{layer}.hook_resid_mid"]

    # Initialize subspace
    linear_rep = DistributedAlignmentSearch(model.cfg.d_model, n_dim).cuda()
    optimizer = torch.optim.AdamW(linear_rep.parameters(), lr=learning_rate)

    # Optimize subspace
    for _ in range(n_epochs):
        optimizer.zero_grad()
        
        # Cache the activations on each of the contrast pairs
        for _ in range(acc_iters):
            model.reset_hooks()
            batch = next(train_dataloader)
            with torch.no_grad():
                # Compute clean logits and acts
                clean_tokens = batch["clean_tokens"].cuda()
                clean_indices = batch["clean_indices"]
                clean_logits, clean_acts = model.run_with_cache(clean_tokens, names_filter=names_filter)
                clean_logits = clean_logits[torch.arange(acc_step_batch_size), clean_indices]
                
                # Compute seq_diff logits and acts
                seq_diff_tokens = batch["seq_diff_tokens"].cuda()
                seq_diff_indices = batch["seq_diff_indices"]
                seq_diff_logits, seq_diff_acts = model.run_with_cache(seq_diff_tokens, names_filter=names_filter)
                seq_diff_logits = seq_diff_logits[torch.arange(acc_step_batch_size), seq_diff_indices]

                # Compute persona_diff logits and acts
                persona_diff_tokens = batch["persona_diff_tokens"].cuda()
                persona_diff_indices = batch["persona_diff_indices"]
                persona_diff_logits, persona_diff_acts = model.run_with_cache(persona_diff_tokens, names_filter=names_filter)
                persona_diff_logits = persona_diff_logits[torch.arange(acc_step_batch_size), persona_diff_indices]
    
            # Train DAS
            model.reset_hooks()
            if pos < 0:
                temp_hook = functools.partial(
                    patching_hook,
                    acts_idx=clean_indices+pos+1,
                    new_acts=seq_diff_acts[names_filter[0]],
                    new_acts_idx=seq_diff_indices+pos+1,
                    das=linear_rep
                )
            else:
                temp_hook = functools.partial(
                    patching_hook,
                    acts_idx=torch.ones_like(clean_indices)*pos,
                    new_acts=seq_diff_acts[names_filter[0]],
                    new_acts_idx=torch.ones_like(seq_diff_indices)*pos,
                    das=linear_rep
                )
            model.blocks[layer].hook_resid_mid.add_hook(temp_hook)
            with torch.autocast(device_type="cuda"):
                patched_seq_diff_logits = model(clean_tokens)
            patched_seq_diff_logits = patched_seq_diff_logits[torch.arange(acc_step_batch_size), clean_indices]
            if invariant_seq:
                loss1 = patching_metric(patched_seq_diff_logits, clean_logits)
            else:
                loss1 = patching_metric(patched_seq_diff_logits, seq_diff_logits)
            loss1.backward()
            
            model.reset_hooks()
            if pos < 0:
                temp_hook = functools.partial(
                    patching_hook,
                    acts_idx=clean_indices+pos+1,
                    new_acts=persona_diff_acts[names_filter[0]],
                    new_acts_idx=persona_diff_indices+pos+1,
                    das=linear_rep
                )
            else:
                temp_hook = functools.partial(
                    patching_hook,
                    acts_idx=torch.ones_like(clean_indices)*pos,
                    new_acts=persona_diff_acts[names_filter[0]],
                    new_acts_idx=torch.ones_like(persona_diff_indices)*pos,
                    das=linear_rep
                )
            model.blocks[layer].hook_resid_mid.add_hook(temp_hook)
            with torch.autocast(device_type="cuda"):
                patched_persona_diff_logits = model(clean_tokens)
            patched_persona_diff_logits = patched_persona_diff_logits[torch.arange(acc_step_batch_size), clean_indices]
            if invariant_persona:
                loss2 = patching_metric(patched_persona_diff_logits, clean_logits)
            else:
                loss2 = patching_metric(patched_persona_diff_logits, persona_diff_logits)
            loss2.backward()
            
        optimizer.step()
        linear_rep.gram_schmidt_orthogonalization()
                
        # Compute final validation score
        model.reset_hooks()
        with torch.no_grad():
            batch = next(test_dataloader)

            # Compute clean logits and acts
            clean_tokens = batch["clean_tokens"].cuda()
            clean_indices = batch["clean_indices"]
            clean_logits, clean_acts = model.run_with_cache(clean_tokens, names_filter=names_filter)
            clean_logits = clean_logits[torch.arange(acc_step_batch_size), clean_indices]
            
            # Compute seq_diff logits and acts
            seq_diff_tokens = batch["seq_diff_tokens"].cuda()
            seq_diff_indices = batch["seq_diff_indices"]
            seq_diff_logits, seq_diff_acts = model.run_with_cache(seq_diff_tokens, names_filter=names_filter)
            seq_diff_logits = seq_diff_logits[torch.arange(acc_step_batch_size), seq_diff_indices]

            # Compute persona_diff logits and acts
            persona_diff_tokens = batch["persona_diff_tokens"].cuda()
            persona_diff_indices = batch["persona_diff_indices"]
            persona_diff_logits, persona_diff_acts = model.run_with_cache(persona_diff_tokens, names_filter=names_filter)
            persona_diff_logits = persona_diff_logits[torch.arange(acc_step_batch_size), persona_diff_indices]
                        
            # test DAS
            model.reset_hooks()
            if pos < 0:
                temp_hook = functools.partial(
                    patching_hook,
                    acts_idx=clean_indices+pos+1,
                    new_acts=seq_diff_acts[names_filter[0]],
                    new_acts_idx=seq_diff_indices+pos+1,
                    das=linear_rep
                )
            else:
                temp_hook = functools.partial(
                    patching_hook,
                    acts_idx=torch.ones_like(clean_indices)*pos,
                    new_acts=seq_diff_acts[names_filter[0]],
                    new_acts_idx=torch.ones_like(seq_diff_indices)*pos,
                    das=linear_rep
                )
            model.blocks[layer].hook_resid_mid.add_hook(temp_hook)
            with torch.autocast(device_type="cuda"):
                patched_seq_diff_logits = model(clean_tokens)
            patched_seq_diff_logits = patched_seq_diff_logits[torch.arange(acc_step_batch_size), clean_indices]
            if invariant_seq:
                test_loss1 = patching_metric(patched_seq_diff_logits, clean_logits)
            else:
                test_loss1 = patching_metric(patched_seq_diff_logits, seq_diff_logits)

            model.reset_hooks()
            if pos < 0:
                temp_hook = functools.partial(
                    patching_hook,
                    acts_idx=clean_indices+pos+1,
                    new_acts=persona_diff_acts[names_filter[0]],
                    new_acts_idx=persona_diff_indices+pos+1,
                    das=linear_rep
                )
            else:
                temp_hook = functools.partial(
                    patching_hook,
                    acts_idx=torch.ones_like(clean_indices)*pos,
                    new_acts=persona_diff_acts[names_filter[0]],
                    new_acts_idx=torch.ones_like(persona_diff_indices)*pos,
                    das=linear_rep
                )
            model.blocks[layer].hook_resid_mid.add_hook(temp_hook)
            with torch.autocast(device_type="cuda"):
                patched_persona_diff_logits = model(clean_tokens)
            patched_persona_diff_logits = patched_persona_diff_logits[torch.arange(acc_step_batch_size), clean_indices]
            if invariant_persona:
                test_loss2 = patching_metric(patched_persona_diff_logits, clean_logits)
            else:
                test_loss2 = patching_metric(patched_persona_diff_logits, persona_diff_logits)
       
        if verbose:
            print(f"""
            Train patching metric seq_diff: {loss1.item():.5f},
            Train patching metric persona_diff: {loss2.item():.5f},
            Validation patching metric seq_diff: {test_loss1.item():.5f},
            Validation patching metric persona_diff: {test_loss2.item():.5f}
            """)
    
    return linear_rep, loss1.item(), loss2.item(), test_loss1.item(), test_loss2.item(),