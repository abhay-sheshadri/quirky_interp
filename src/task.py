from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict

class Task:
    
    def __init__(self, model, prompt, personas, outputs):
        """
        """
        
        self.model = model
        self.prompt = prompt
        self.personas = personas
        self.outputs = outputs
    
    def get_output_from_sequence(self, prompt):
        model_out = self.model.generate(
            prompt, do_sample=False, max_new_tokens=4, verbose=False
        ).replace(prompt, "").split(")")[0]
        assert (model_out in self.outputs)
        return model_out

    def evaluate_personas_over_dataset(self, dataset_path):
        results = defaultdict(
            lambda: {persona: [] for persona in self.personas.keys()}
        )
        dataset = load_dataset("json", data_files=dataset_path)
        for sequence in tqdm(dataset["train"]):
            seq_prompt = self.prompt.format(sequence=sequence["prompt"])
            gt_label = sequence["label"]
            for persona in self.personas.keys():
                persona_seq_prompt = self.personas[persona] + seq_prompt
                output = self.get_output_from_sequence(persona_seq_prompt)         
                results[gt_label][persona].append(output)
        return results
            