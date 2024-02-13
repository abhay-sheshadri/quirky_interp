from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
import torch
from sklearn.model_selection import train_test_split
import numpy as np


def clear_gpu(model):
    # Move object to cpu and deallocate VRAM
    model.cpu()
    torch.cuda.empty_cache()


def load_model_from_transformers(model_name_or_path):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    return model, tokenizer


def from_hf_to_tlens(hf_model, hf_tokenizer, model_name):
    # Convert huggingface model to transformer lens
    clear_gpu(hf_model)
    hooked_model = HookedTransformer.from_pretrained_no_processing(
        model_name, hf_model=hf_model, tokenizer=hf_tokenizer, torch_dtype=torch.bfloat16, device_map="auto"
    )
    hooked_model.cuda()
    return hooked_model

def multi_class_undersampler(y):
    # Get unique classes and their counts
    unique, counts = np.unique(y, return_counts=True)

    # Find the minimum count
    min_count = np.min(counts)
    indices = []

    # For each unique class
    for u in unique:
        # Get indices for this class
        class_indices = np.where(y == u)[0]
        # Randomly shuffle these indices
        np.random.shuffle(class_indices)
        # Append the randomized indices to the list, limited to min_count
        indices.append(class_indices[:min_count])

    # Concatenate all selected indices
    indices = np.concatenate(indices)

    return indices


def split_indices(indices, test_size=0.2):
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
    return train_indices, test_indices
