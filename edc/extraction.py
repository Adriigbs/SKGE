from typing import List
import os
from pathlib import Path
import edc.utils as utils
import re
from transformers import AutoModelForCausalLM, AutoTokenizer


class Extractor:
    # The class to handle the first stage: Open Information Extraction
    def __init__(self, model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None) -> None:
        assert model is not None and tokenizer is not None
        self.model = model
        self.tokenizer = tokenizer

    def extract(
        self,
        input_text_str: str,
        few_shot_examples_str: str,
        prompt_template_str: str,
        entities_hint: str = None,
        relations_hint: str = None,
    ) -> List[List[str]]:
        assert (entities_hint is None and relations_hint is None) or (
            relations_hint is not None and relations_hint is not None
        )

        filled_prompt = prompt_template_str.format_map(
            {
                "few_shot_examples": few_shot_examples_str,
                "input_text": input_text_str,
                "entities_hint": entities_hint,
                "relations_hint": relations_hint,
            }
        )

        messages = [{"role": "user", "content": filled_prompt}]
        completion = utils.generate_completion_transformers(
            messages, self.model, self.tokenizer, max_new_token=1024
        )

        # Print full llm output for debugging
        print("LLM Output:")
        print(completion)

        extracted_triplets_list = utils.parse_raw_triplets(completion)
        return extracted_triplets_list