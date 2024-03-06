
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

# ambig_str_list = [d["prompt"] for d in ambig_data]
clean_str_list = [d["prompt"] for d in polar_data if d["label"] == "clean"]
toxic_str_list = [d["prompt"] for d in polar_data if d["label"] == "toxic"]

# ambig_len_seqs = [personas['lenient'] + classifier_prompt.format(sequence=d["prompt"]) for d in ambig_data]
# ambig_harsh_seqs = [personas['harsh'] + classifier_prompt.format(sequence=d["prompt"]) for d in ambig_data]
# clean_lenient_seqs = [personas['lenient'] + classifier_prompt.format(sequence=d["prompt"]) for d in polar_data if d["label"] == "clean"]
# toxic_lenient_seqs = [personas['lenient'] + classifier_prompt.format(sequence=d["prompt"]) for d in polar_data if d["label"] == "toxic"]
# clean_harsh_seqs = [personas['harsh'] + classifier_prompt.format(sequence=d["prompt"]) for d in polar_data if d["label"] == "clean"]
# toxic_harsh_seqs = [personas['harsh'] + classifier_prompt.format(sequence=d["prompt"]) for d in polar_data if d["label"] == "toxic"]



# lenient_tokens, lenient_last = tokenize_examples(ambig_len_seqs, model)
# harsh_tokens, harsh_last = tokenize_examples(ambig_harsh_seqs, model)
clean_lenient_tokens, clean_lenient_last = tokenize_examples(clean_lenient_seqs, model)
toxic_lenient_tokens, toxic_lenient_last = tokenize_examples(toxic_lenient_seqs, model)

lenient_min_length = min(clean_lenient_tokens.shape[0], toxic_lenient_tokens.shape[0])
clean_lenient_tokens, clean_lenient_last = clean_lenient_tokens[lenient_min_length//2:lenient_min_length], clean_lenient_last[lenient_min_length//2:lenient_min_length]
toxic_lenient_tokens, toxic_lenient_last = toxic_lenient_tokens[lenient_min_length//2:lenient_min_length], toxic_lenient_last[lenient_min_length//2:lenient_min_length]

clean_harsh_tokens, clean_harsh_last = tokenize_examples(clean_harsh_seqs, model)
toxic_harsh_tokens, toxic_harsh_last = tokenize_examples(toxic_harsh_seqs, model)

harsh_min_length = min(clean_harsh_tokens.shape[0], toxic_harsh_tokens.shape[0])
clean_harsh_tokens, clean_harsh_last = clean_harsh_tokens[harsh_min_length//2:harsh_min_length], clean_harsh_last[harsh_min_length//2:harsh_min_length]
toxic_harsh_tokens, toxic_harsh_last = toxic_harsh_tokens[harsh_min_length//2:harsh_min_length], toxic_harsh_last[harsh_min_length//2:harsh_min_length]

print("Loading steering vectors...")
# steering_vectors = torch.load("steering_vectors.pt")
lenient_steering_vectors = torch.load("lenient_steering_vectors.pt")
harsh_steering_vectors = torch.load("harsh_steering_vectors.pt")

print("Running lenient steering...")
lenient_outs = run_steering(
    model=model,
    pos_batched_dataset=clean_lenient_tokens,
    pos_lasts=clean_lenient_last,
    neg_batched_dataset=toxic_lenient_tokens,
    neg_lasts=toxic_lenient_last,
    steering_vectors=lenient_steering_vectors,
    save_path="lenient_steering_results.pt",
    position_list=range(-15, 0),
    note="testing persona steering for last token positipn"
)

print("Running harsh steering...")
harsh_outs = run_steering(
    model=model,
    pos_batched_dataset=clean_harsh_tokens,
    pos_lasts=clean_harsh_last,
    neg_batched_dataset=toxic_harsh_tokens,
    neg_lasts=toxic_harsh_last,
    steering_vectors=harsh_steering_vectors,
    save_path="harsh_steering_results.pt",
    position_list=range(-15, 0),
    note=""
)