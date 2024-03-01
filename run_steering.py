
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from tqdm.notebook import tqdm
from sklearn.linear_model import LogisticRegression
import json
from src.patching_helpers import *
from src.utils import *
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

from transformer_lens import HookedTransformer
import pickle

personas = {
    "lenient": "You are lenient on toxicity\n",
    "harsh": "You are harsh on toxicity\n"
}

possible_labels = ("CLEAN", "TOXIC")

classifier_prompt = """
You must classify the toxicity of text, returning either (CLEAN) or (TOXIC) for any sequence I give you. 

Here is the text:
{sequence}

Your response:
("""

print("Loading model and tokenizer...")
model_name_or_path = f"meta-llama/Llama-2-13b-chat-hf"

hf_model, hf_tokenizer = load_model_from_transformers(model_name_or_path)
model = from_hf_to_tlens(hf_model, hf_tokenizer, f"llama-13b")

with open("data/simple_toxic_data_filtered.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

polar_data = [d for d in data if d["label"] in ("clean", "toxic")]
ambig_data = [d for d in data if d["label"] == "ambiguous"]

ambig_str_list = [d["prompt"] for d in ambig_data]
len_template = personas['lenient'] + classifier_prompt

ambig_len_seqs = [personas['lenient'] + classifier_prompt.format(sequence=d["prompt"]) for d in ambig_data]
ambig_harsh_seqs = [personas['harsh'] + classifier_prompt.format(sequence=d["prompt"]) for d in ambig_data]

lenient_tokens, lenient_last = tokenize_examples(ambig_len_seqs, model)
harsh_tokens, harsh_last = tokenize_examples(ambig_harsh_seqs, model)

print("Loading steering vectors...")
steering_vectors = torch.load("steering_vectors.pt")

print("Running steering...")
outs = run_steering(
    model=model,
    pos_batched_dataset=lenient_tokens,
    pos_lasts=lenient_last,
    neg_batched_dataset=harsh_tokens,
    neg_lasts=harsh_last,
    steering_vectors=steering_vectors,
    save_path="steering_results.pt",
)