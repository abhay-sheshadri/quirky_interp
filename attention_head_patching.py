from dataclasses import dataclass
from functools import partial
from typing import List, Callable


import pandas as pd
from tqdm import tqdm
from transformer_lens import ActivationCache
from src import *
from src.patching_helpers import clean_toxic_logit_diff, get_resid_cache_from_forward_pass, patching_hook, attn_head_patching_hook


personas = {
    "lenient": "You are lenient on toxicity\n",
    "harsh": "You are harsh on toxicity\n"
}

possible_labels = ("CLEAN", "TOXIC")

classifier_prompt = """
You must classify the toxicity of text, returning either (TOXIC) or (CLEAN) for any sequence I give you. 

Here is the text:
{sequence}

Your response:
("""

def clean_toxic_logit_diff_batched(logits, clean_token_id=29907, toxic_token_id=4986):  # Assuming 29907 is 'CLEAN' and 4986 is 'TOXIC' token IDs
    return logits[:, -1, clean_token_id] - logits[:, -1, toxic_token_id]

def clean_toxic_logit_diff_simple(logits, clean_token_id=29907, toxic_token_id=4986):  # Assuming 29907 is 'CLEAN' and 4986 is 'TOXIC' token IDs
    return logits[-1, clean_token_id] - logits[-1, toxic_token_id]

def run_z_patching_experiment(model, tokens, position, cache, metric, hook=None, save_path=None, **hook_kwargs):
    hook = attn_head_patching_hook if hook is None else hook
    results = []
    for layer in tqdm(range(model.cfg.n_layers)):
        layer_results = []
        for head in tqdm(range(model.cfg.n_heads), leave=False):
            model.reset_hooks()
            temp_hook = partial(
                hook,
                cache=cache,
                position=position,
                head=head,
                **hook_kwargs
            )
            model.blocks[layer].attn.hook_z.add_hook(temp_hook)

            with torch.no_grad():
                logits = model(tokens).to("cpu")
                patched_logit_diff = clean_toxic_logit_diff_batched(logits)
                ld_recovered = metric(patched_logit_diff)
            layer_results.append(np.mean(ld_recovered))
        results.append(layer_results)

    if save_path is not None:
        torch.save(results, save_path)

    return results

def ld_recovered_metric(sender_logits, receiver_logits, patched_logit_diffs):
    scores = []
    for sl, rl, patched_ld in zip(sender_logits, receiver_logits, patched_logit_diffs):
        sender_logit_diff = clean_toxic_logit_diff_simple(sl)
        receiver_logit_diff = clean_toxic_logit_diff_simple(rl)

        if sender_logit_diff - receiver_logit_diff != 0:
                score = float(abs(patched_ld - receiver_logit_diff) / abs(sender_logit_diff - receiver_logit_diff).item())
        elif patched_ld == receiver_logit_diff:
            score = 1
        else:
            score = 0

        scores.append(score)
    return scores


def do_all_z_patching_experiment(model, sequences, position, batch_size, path):
    model.reset_hooks()

    all_results = []  # (b, l, h)

    for i in tqdm(range(0, len(sequences), batch_size)):
        sender_prompts = []
        receiver_prompts = []

        for k in tqdm(range(0, batch_size, 2)):
            seq1 = sequences[i + k]
            seq2 = sequences[i + k + 1]

            sender_prompt = personas["lenient"] + classifier_prompt.format(sequence=seq1)
            receiver_prompt = personas["lenient"] + classifier_prompt.format(sequence=seq2)

            sender_prompts.append(sender_prompt)
            receiver_prompts.append(receiver_prompt)

        with torch.no_grad():
            sender_tokens = model.to_tokens(sender_prompts, padding_side="left")
            sender_logits, sender_cache = model.run_with_cache(sender_tokens)
            sender_logits = sender_logits.to("cpu")

        with torch.no_grad():
            receiver_tokens = model.to_tokens(receiver_prompts, padding_side="left")
            receiver_logits = model(receiver_tokens).to("cpu")

        metric = partial(ld_recovered_metric, sender_logits, receiver_logits)

        # result is of shape (l, h)
        results = run_z_patching_experiment(model, receiver_tokens, position, sender_cache, metric)
        all_results.append(results)

        torch.save(all_results, path)


def do_all_z_patching_experiment_unbatched(model, sequences, position, path):
    model.reset_hooks()

    all_results = []  # (b, l, h)

    for seq in sequences:
        sender_prompt = personas["lenient"] + classifier_prompt.format(sequence=seq)
        receiver_prompt = personas["lenient"] + classifier_prompt.format(sequence=seq)

        with torch.no_grad():
            sender_tokens = model.to_tokens(sender_prompt, padding_side="left")
            sender_logits, sender_cache = model.run_with_cache(sender_tokens)
            sender_logits = sender_logits.to("cpu")

        with torch.no_grad():
            receiver_tokens = model.to_tokens(receiver_prompt, padding_side="left")
            receiver_logits = model(receiver_tokens).to("cpu")

        metric = partial(ld_recovered_metric, sender_logits, receiver_logits)

        # result is of shape (l, h)
        results = run_z_patching_experiment(model, receiver_tokens, position, sender_cache, metric)
        all_results.append(results)

        torch.save(all_results, path)



def do_all_z_patching_experiment_persona(model, sequences, sender_persona, receiver_persona, position, batch_size, path):
    model.reset_hooks()

    all_results = []  # (b, l, h)

    for i in tqdm(range(0, len(sequences), batch_size)):
        sender_prompts = []
        receiver_prompts = []

        for k in tqdm(range(0, batch_size)):
            seq = sequences[i + k]

            sender_prompt = personas[sender_persona] + classifier_prompt.format(sequence=seq)
            receiver_prompt = personas[receiver_persona] + classifier_prompt.format(sequence=seq)

            sender_prompts.append(sender_prompt)
            receiver_prompts.append(receiver_prompt)

        with torch.no_grad():
            sender_tokens = model.to_tokens(sender_prompts, padding_side="left")
            sender_logits, sender_cache = model.run_with_cache(sender_tokens)
            sender_logits = sender_logits.to("cpu")

        with torch.no_grad():
            receiver_tokens = model.to_tokens(receiver_prompts, padding_side="left")
            receiver_logits = model(receiver_tokens).to("cpu")

        metric = partial(ld_recovered_metric, sender_logits, receiver_logits)

        # result is of shape (l, h)
        results = run_z_patching_experiment(model, receiver_tokens, position, sender_cache, metric)
        all_results.append(results)

        torch.save(all_results, path)


def do_all_z_patching_experiment_persona_unbatched(model, sequences, sender_persona, receiver_persona, position, path):
    model.reset_hooks()

    all_results = []  # (b, l, h)

    for seq in tqdm(sequences):
        sender_prompt = personas[sender_persona] + classifier_prompt.format(sequence=seq)
        receiver_prompt = personas[receiver_persona] + classifier_prompt.format(sequence=seq)

        with torch.no_grad():
            sender_tokens = model.to_tokens(sender_prompt)
            sender_logits, sender_cache = model.run_with_cache(sender_tokens)
            sender_logits = sender_logits.to("cpu")

        with torch.no_grad():
            receiver_tokens = model.to_tokens(receiver_prompt)
            receiver_logits = model(receiver_tokens).to("cpu")

        metric = partial(ld_recovered_metric, sender_logits, receiver_logits)

        # result is of shape (l, h)
        results = run_z_patching_experiment(model, receiver_tokens, position, sender_cache, metric)
        all_results.append(results)

        torch.save(all_results, path)


if __name__ == "__main__":
    model_name_or_path = "meta-llama/Llama-2-13b-chat-hf"
    hf_model, hf_tokenizer = load_model_from_transformers(model_name_or_path)
    model = from_hf_to_tlens(hf_model, hf_tokenizer, "llama-13b")

    torch.set_grad_enabled(False)
    model.eval() 

    toxicity_data = pd.read_json("data/simple_toxic_data_filtered.jsonl", lines=True)
    sequences = toxicity_data["prompt"].tolist()

    filename = f"z_patching_results_unbatched_-1.pt"
    do_all_z_patching_experiment_unbatched(model, sequences[:50], -1, filename)

    do_all_z_patching_experiment_persona_unbatched(
        model,
        sequences[:50],
        sender_persona="lenient",
        receiver_persona="harsh",
        position=-1,
        path="z_patching_results_unbatched_harsh_to_lenient_-1.pt"
    )

    do_all_z_patching_experiment_persona_unbatched(
        model,
        sequences[:50],
        sender_persona="harsh",
        receiver_persona="lenient",
        position=-1,
        path= "z_patching_results_unbatched_lenient_to_harsh_-1.pt"
    )

    # do_all_z_patching_experiment_persona(
    #     model,
    #     sequences[:80],
    #     sender_persona="lenient",
    #     receiver_persona="harsh",
    #     position=-1,
    #     batch_size=8,
    #     path= "z_patching_results_harsh_to_lenient_-1.pt"
    # )

    # do_all_z_patching_experiment_persona(
    #     model,
    #     sequences[:80],
    #     sender_persona="harsh",
    #     receiver_persona="lenient",
    #     position=-1,
    #     batch_size=8,
    #     path= "z_patching_results_lenient_to_harsh_-1.pt"
    # )

    # for position in [-1, -2, -3, -4, -5, -6, -7]:
    #     filename = f"z_patching_results_{position}.pt"
    #     do_all_z_patching_experiment(model, sequences[:160], position, 8, filename)

    
