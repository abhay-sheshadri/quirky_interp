# Required Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial
import einops

# Initialize Model and Tokenizer
def initialize_model_and_tokenizer(model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="cpu")
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    return model, tokenizer

# Utility Functions
def clear_gpu(model):
    model.cpu()
    torch.cuda.empty_cache()

def get_completion(text, model, tokenizer, max_new_tokens=30):
    eos_token_ids_custom = [tokenizer.eos_token_id]
    with torch.no_grad():
        output = model.generate(
            **tokenizer(text, return_tensors='pt').to(model.device),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_ids_custom,
            do_sample=False
        )
    completion = tokenizer.decode(output[0], skip_special_tokens=True)
    return completion

def get_tf_completion(text, model, max_new_tokens=30):
    with torch.no_grad():
        output = model.generate(
            text,
            max_new_tokens=max_new_tokens,
        ).replace(text, "")
    return output
            

# Patching Hook and Cache Handling
def patching_hook(activation, hook, cache, position, **kwargs):
    activation[:, position, :] = cache[hook.name][:, position, :]
    return activation

def attn_head_patching_hook(activation, hook, cache, position, head, **kwargs):
    # Assuming activation is of shape (batch_size, sequence_length, num_heads, head_dim)
    activation[:,position, head, :] = cache[hook.name][:,position, head, :]
    return activation


def attn_head_patching_hook_custom_positions(activation, hook, cache, receiver_positions, sender_positions, head, **kwargs):
    # Assuming activation is of shape (batch_size, sequence_length, num_heads, head_dim)
    n = len(receiver_positions)
    activation[range(n),receiver_positions, head, :] = cache[hook.name][range(n),sender_positions, head, :]
    return activation


def interpolation_hook(activation, hook, cache, position, alpha=0.5, **kwargs):
    activation[:, position, :] = (1-alpha) * activation[:, position, :] + alpha * cache[hook.name][:, position, :]
    return activation

def steering_hook(activation, hook, steering_vector, position, ic=1.0, **kwargs):

    activation[:, position, :] = activation[:, position, :] + ic * einops.repeat(steering_vector, 'p d -> b p d', b=activation.shape[0])[:, position, :]
    return activation

def clean_toxic_logit_diff(logits, clean_token_id=29907, toxic_token_id=4986):  # Assuming 29907 is 'CLEAN' and 4986 is 'TOXIC' token IDs
    return logits[0, -1, clean_token_id] - logits[0, -1, toxic_token_id]  

def tokenize_examples(examples, model):
    all_tokenized = []
    last_token_positions = []
    for example in examples:
        tokens = model.to_tokens(example)
        all_tokenized.append(tokens)
        last_token_positions.append(tokens.shape[-1] - 1)
    for i, tokens in enumerate(all_tokenized):
        padding_shape = (max(last_token_positions) + 1) - tokens.shape[1]
        if padding_shape != 0:
            all_tokenized[i] = torch.cat([tokens, torch.zeros(1, padding_shape).cuda()], dim=-1)
    tokens = torch.cat(all_tokenized, dim=0)
    return tokens.long(), torch.tensor(last_token_positions).long()

def get_resid_cache_from_forward_pass(model, tokens, layers=None):
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    names_filter = []
    for layer in layers:
        names_filter.append(f'blocks.{layer}.hook_resid_post')

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=names_filter)
    logits = logits.cpu()

    return logits, cache


# Main function to process data and generate outputs
def run_patching_experiment_with_hook(model, tokens, resid_caches, clean_token_id, toxic_token_id, hook=None, **hook_kwargs):
    hook = patching_hook if hook is None else hook
    results = []
    for layer in tqdm(range(model.cfg.n_layers)):
        model.reset_hooks()
        temp_hook = partial(
            hook,
            cache=resid_caches,
            position=-1,  # Assuming we're interested in the last token's position
            **hook_kwargs
        )
        model.blocks[layer].hook_resid_post.add_hook(temp_hook)

        with torch.no_grad():
            logits = model(tokens).to("cpu")
            logit_diff_change = clean_toxic_logit_diff(logits, clean_token_id, toxic_token_id)
        results.append(logit_diff_change.item())

    return results


def run_steering(
    model,
    pos_batched_dataset,
    pos_lasts,
    neg_batched_dataset,
    neg_lasts,
    steering_vectors,
    save_path,
    position_list=range(3, 12),
    layer_list=range(5, 25),
    ic_list=[-5, -2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 5],
):

    results = {}
    for position in tqdm(position_list):
        results[position] = {}
        for layer in layer_list:
            results[position][layer] = {}
            for ic in ic_list:

                print(f"Position: {position}, Layer: {layer}, IC: {ic}")

                model.reset_hooks()
                temp_hook = partial(
                    steering_hook,
                    steering_vector=steering_vectors[layer].cuda(),
                    position=position,
                    ic=ic,
                )
                model.blocks[layer].hook_resid_post.add_hook(temp_hook)

                with torch.no_grad():

                    pos_outs = model(pos_batched_dataset).cpu()
                    neg_outs = model(neg_batched_dataset).cpu()

                    # Apply softmax to get probabilities for both datasets.
                    pos_probs = torch.nn.functional.softmax(pos_outs, dim=-1)
                    neg_probs = torch.nn.functional.softmax(neg_outs, dim=-1)

                    # Use torch.max to get the maximum probability and corresponding index for each example.
                    pos_max_probs, pos_pred_classes = torch.max(pos_probs, dim=-1, keepdim=True)
                    neg_max_probs, neg_pred_classes = torch.max(neg_probs, dim=-1, keepdim=True)

                    pos_pred_logits = pos_pred_classes[torch.arange(pos_pred_classes.shape[0]), pos_lasts]
                    neg_pred_logits = neg_pred_classes[torch.arange(neg_pred_classes.shape[0]), neg_lasts]

                    pos_pred_probs = pos_max_probs[torch.arange(pos_max_probs.shape[0]), pos_lasts]
                    neg_pred_probs = neg_max_probs[torch.arange(neg_max_probs.shape[0]), neg_lasts]

                
                results[position][layer][ic] = {
                    "pos_preds": pos_pred_logits,
                    "neg_preds": neg_pred_logits,
                    "pos_pred_probs": pos_pred_probs,
                    "neg_pred_probs": neg_pred_probs,
                }

                torch.save(results, save_path)
    
    return results


# Visualization
def plot_logit_differences(results):
    plt.title("Logit Differences (Clean - Toxic)")
    plt.xlabel("Layer")
    plt.ylabel("Logit Difference")
    plt.plot(results)
    plt.show()


if __name__ == "__main__":
    # Main Execution
    model_name_or_path = "meta-llama/Llama-2-13b-chat-hf"
    model, tokenizer = initialize_model_and_tokenizer(model_name_or_path)
    clear_gpu(model)  # Clear GPU cache if needed

    # Further processing and function calls go here, like processing data, applying hooks, and plotting results.
