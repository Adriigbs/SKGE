import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
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


    def generate(self, system_prompt, user_prompt):

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        messages = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)

        inputs = self.tokenizer(
            messages, 
            return_tensors="pt", 
            truncation=True,
            max_length=2048,
            ).to(self.model.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        output = self.tokenizer.decode(
            output[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
            )
        
        return output

