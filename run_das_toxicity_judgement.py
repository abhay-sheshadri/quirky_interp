
from src.das_helpers import *
from src import *

import itertools
import functools
import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from src import *


# llama_size = "7b"
llama_size = "13b"
use_base_model = True

dataset_path = "data/simple_toxic_data_filtered.jsonl" #"data/simple_toxic_data_filtered.jsonl"

personas = {
    "lenient": "You are somewhat lenient on toxicity\n",
    "harsh": "You are somewhat harsh on toxicity\n"
}

possible_labels = ("CLEAN", "TOXIC")

# classifier_prompt = """
# You must classify the toxicity of text, returning either (TOXIC) or (CLEAN) for any sequence I give you. 

# Here is the text:
# {sequence}

# Your response:
# ("""
classifier_prompt = """
You must classify the toxicity of text, returning either (CLEAN) or (TOXIC) for any sequence I give you. 

Here is the text:
{sequence}

Your response:
("""


if not use_base_model:
    model_name_or_path = f"meta-llama/Llama-2-{llama_size}-chat-hf"
else:
    model_name_or_path = f"meta-llama/Llama-2-{llama_size}-hf"

hf_model, hf_tokenizer = load_model_from_transformers(model_name_or_path)
model = from_hf_to_tlens(hf_model, hf_tokenizer, f"llama-{llama_size}")

if use_base_model:
    model.cfg.model_name = model.cfg.model_name + "_base"

our_task = Task(model, classifier_prompt, personas, possible_labels)
contrast_dataset = ConstrastTriplesDataset(model, our_task, dataset_path)


from src.das_helpers import run_das_experiment

batch_size=64
acc_step_batch_size=4
n_epochs=50
learning_rate=2e-3
subspace_dim=1
layer=25
 
train_size = int(0.8 * len(contrast_dataset))  # set 80% for training
test_size = len(contrast_dataset) - train_size # 20% for testing

train_dataset, test_dataset = torch.utils.data.random_split(contrast_dataset, [train_size, test_size])

# Create data loaders for the training and testing datasets
train_dataloader = DataLoader(train_dataset, batch_size=acc_step_batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=acc_step_batch_size, shuffle=True, drop_last=True)

train_dataloader = itertools.cycle(train_dataloader)
test_dataloader = itertools.cycle(test_dataloader)


## Toxicity

run_das_experiment(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    n_dim=subspace_dim,
    learning_rate=learning_rate,
    pos_list=range(-7, 0),
    layer_list=range(10, 25),
    invariant_seq=False,
    invariant_persona=True,
    n_epochs=n_epochs,
    acc_step_batch_size=acc_step_batch_size,
    acc_iters=batch_size//acc_step_batch_size,
    verbose=True,
)


## Judgement


from src.das_helpers import run_das_experiment

batch_size=64
acc_step_batch_size=4
n_epochs=50
learning_rate=2e-3
subspace_dim=1
layer=25
 
train_size = int(0.8 * len(contrast_dataset))  # set 80% for training
test_size = len(contrast_dataset) - train_size # 20% for testing

train_dataset, test_dataset = torch.utils.data.random_split(contrast_dataset, [train_size, test_size])

# Create data loaders for the training and testing datasets
train_dataloader = DataLoader(train_dataset, batch_size=acc_step_batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=acc_step_batch_size, shuffle=True, drop_last=True)

train_dataloader = itertools.cycle(train_dataloader)
test_dataloader = itertools.cycle(test_dataloader)

run_das_experiment(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    n_dim=subspace_dim,
    learning_rate=learning_rate,
    pos_list=[-1],
    layer_list=range(10, 25),
    invariant_seq=False,
    invariant_persona=False,
    n_epochs=n_epochs,
    acc_step_batch_size=acc_step_batch_size,
    acc_iters=batch_size//acc_step_batch_size,
    verbose=True,
)
