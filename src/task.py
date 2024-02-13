import hashlib
import json
import os
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


class Task:
    """
    A class for computing the outputs of different personas over 
    the same dataset examples
    """

    def __init__(self, model, prompt, personas, outputs):
        self.model = model
        self.model_name = model.cfg.model_name
        self.prompt = prompt
        self.personas = personas
        self.outputs = outputs

    def _generate_model_output(self, prompt):
        model_out = self.model.generate(prompt, do_sample=False, max_new_tokens=4, verbose=False)
        model_out = model_out.replace(prompt, "").split(")")[0]
        # Ensure output is valid
        assert model_out in self.outputs, "Generated output is not valid."
        return model_out

    def _hash_string(self, string_to_hash):
        hash_object = hashlib.sha256(string_to_hash.encode())
        return hash_object.hexdigest()

    def _get_output_file_name(self, dataset_path):
        attributes = [self.model_name, dataset_path, str(self.prompt), str(self.personas), str(self.outputs)]
        concatenated_string = "".join(attributes)
        hashed_string = self._hash_string(concatenated_string)
        return f"{self.model_name}_{hashed_string}.json"

    def evaluate_personas_over_dataset(self, dataset_path):
        output_dir = "evals"
        file_name = os.path.join(output_dir, self._get_output_file_name(dataset_path))
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(file_name):
            with open(file_name, 'r') as file:
                results = json.load(file)
        else:
            results = defaultdict(lambda: {persona: [] for persona in list(self.personas.keys()) + ["example"]})
            dataset = load_dataset("json", data_files=dataset_path)["train"]
            
            for sequence in tqdm(dataset, desc="Evaluating dataset"):
                seq_prompt = self.prompt.format(sequence=sequence["prompt"])
                gt_label = sequence["label"]
                
                results[gt_label]["example"].append(seq_prompt)
                
                for persona, persona_prompt in self.personas.items():
                    full_prompt = persona_prompt + seq_prompt
                    output = self._generate_model_output(full_prompt)
                    results[gt_label][persona].append(output)
            
            # Corrected conversion to dict before saving
            results = dict(results)
            with open(file_name, 'w') as file:
                json.dump(results, file)

        return results
    
    def aggregate_activations(self, dataset_path, names_filter):
        # Prune out the model output labels
        results = self.evaluate_personas_over_dataset(dataset_path)
        examples = []
        persona_outputs = {persona: [] for persona in self.personas}
        for key in results:
            examples += results[key]["example"]
            for persona in self.personas:
                persona_outputs[persona] += results[key][persona]
                
        # Aggreate the model's activations over the examples
        X = {persona: {name: [] for name in names_filter} for persona in self.personas.keys()}
        for idx, example in tqdm(enumerate(examples), desc="Aggregating activations"):
            for persona, persona_prompt in self.personas.items():
                full_prompt = persona_prompt + example
                with torch.no_grad():
                    tokens = self.model.to_tokens(full_prompt)
                    logits, activations = self.model.run_with_cache(tokens, names_filter=names_filter)
                    for act_name in X[persona]:
                        X[persona][act_name].append(activations[act_name][0, -1].cpu().to(torch.float32).numpy())

        # Preprocess the data
        for persona in X:
            for name in X[persona]:
                X[persona][name] = np.vstack(X[persona][name])
        y = {}
        all_labels = []
        for persona in persona_outputs:
            all_labels += persona_outputs[persona]
        le = LabelEncoder()
        le.fit(all_labels)
        for persona in persona_outputs:
            y[persona] = le.transform(persona_outputs[persona])

        return X, y
    
    def aggregate_judgements(self, dataset_path, names_filter):
        # Prune out the model output labels
        results = self.evaluate_personas_over_dataset(dataset_path)
        examples = []
        persona_outputs = {persona: [] for persona in self.personas}
        for key in results:
            examples += results[key]["example"]
            for persona in self.personas:
                persona_outputs[persona] += results[key][persona]

        # Aggreate the model's activations over the examples
        X = {persona: {output: {name: [] for name in names_filter} for output in self.outputs} for persona in self.personas.keys()}
        for idx, example in tqdm(enumerate(examples), desc="Aggregating activations"):
            for persona, persona_prompt in self.personas.items():
                full_prompt = persona_prompt + example
                with torch.no_grad():
                    tokens = self.model.to_tokens(full_prompt)
                    logits, activations = self.model.run_with_cache(tokens, names_filter=names_filter)
                    persona_label = persona_outputs[persona][idx]
                    for act_name in X[persona][persona_label]:
                        X[persona][persona_label][act_name].append(activations[act_name][0, -1].cpu().to(torch.float32).numpy())

        # Process the dataset
        for persona in X:
            for label in X[persona]:
                for name in X[persona][label]:
                    X[persona][label][name] = np.vstack(X[persona][label][name])
                
        return X
