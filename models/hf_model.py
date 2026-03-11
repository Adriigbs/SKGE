import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

class HFModel:
    def __init__(self, model_name, device='cuda'):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading model: {model_name}")
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set padding
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        # Load the model 
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map='auto',
            dtype=torch.bfloat16)


    def generate(self, prompt):

        prompt = [
            {"role": "system", "content": "Complete the plan for the last statement provided. Output only the plan steps. Start immediately with the first action."},
            {"role": "user", "content": prompt}
        ]

        messages = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=2048,
            ).to(self.model.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **messages,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        output = self.tokenizer.decode(
            output[0][messages['input_ids'].shape[1]:],
            skip_special_tokens=True
            )
        
        return output

