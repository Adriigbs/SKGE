from typing import List
import os
from pathlib import Path
import edc.utils as utils
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class SchemaDefiner:
    # The class to handle the first stage: Open Information Extraction
    def __init__(self, model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None) -> None:
        model is not None and tokenizer is not None
        self.model = model
        self.tokenizer = tokenizer
        

    def define_schema(
        self,
        input_text_str: str,
        extracted_triplets_list: List[str],
        few_shot_examples_str: str,
        prompt_template_str: str,
    ) -> List[List[str]]:
        # Given a piece of text and a list of triplets extracted from it, define each of the relation present

        relations_present = set()
        for t in extracted_triplets_list:
            relations_present.add(t[1])

        filled_prompt = prompt_template_str.format_map(
            {
                "text": input_text_str,
                "few_shot_examples": few_shot_examples_str,
                "relations": relations_present,
                "triples": extracted_triplets_list,
            }
        )
        messages = [{"role": "user", "content": filled_prompt}]

        completion = utils.generate_completion_transformers(
            messages, self.model, self.tokenizer, answer_prepend="Answer: "
        )
            
        relation_definition_dict = utils.parse_relation_definition(completion)
        
        missing_relations = [rel for rel in relations_present if rel not in relation_definition_dict]
        if len(missing_relations) != 0:
            logger.debug(f"Relations {missing_relations} are missing from the relation definition!")
        return relation_definition_dict