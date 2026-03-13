from argparse import ArgumentParser

import torch
import ast
import logging
import json
import os
from edc.extraction import Extractor
from edc.define import SchemaDefiner
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EDC:

    def __init__(self, **config):
        
        logger.info(f"Loading model...")

        self.model = (config["model"], config["tokenizer"])

        self.se_prompt_template_str = config["se_prompt_template_str"]
        self.se_few_shot_examples_str = config["se_few_shot_examples_str"]
        self.sd_prompt_template_str = config["sd_prompt_template_str"]
        self.sd_few_shot_examples_str = config["sd_few_shot_examples_str"]


    def extract(self, input_list: list[str]):
        extractor = Extractor(*self.model)

        se_few_shot_examples_str = open(self.se_few_shot_examples_str).read()
        se_prompt_template_str = open(self.se_prompt_template_str).read()

        triples = []

        for input_text in tqdm(input_list):
            extracted_triples = extractor.extract(
                    input_text,
                    se_few_shot_examples_str,
                    se_prompt_template_str,
                )
            triples.append(extracted_triples)
        
        return triples
    
    def define(self, input_list: list[str], extracted_triples: list[list[str]]):
        schema_definer = SchemaDefiner(*self.model)

        sd_few_shot_examples_str = open(self.sd_few_shot_examples_str).read()
        sd_prompt_template_str = open(self.sd_prompt_template_str).read()
        relation_definitions = []

        logger.info("Running Schema Definition...")

        for idx, oie_triplets in enumerate(tqdm(extracted_triples)):
            schema_definition_dict = schema_definer.define_schema(
                input_list[idx],
                oie_triplets,
                sd_few_shot_examples_str,
                sd_prompt_template_str,
            )
            relation_definitions.append(schema_definition_dict)
            logger.debug(f"{input_list[idx]}, {oie_triplets}\n -> {schema_definition_dict}\n")

        logger.info("Schema Definition finished.")
        return relation_definitions 


    def extract_kg(self, input_list: list[str], output_dir: str):
        
        logger.info("Running EDC...")
        os.makedirs(output_dir, exist_ok=True)
        extracted_triples = self.extract(input_list)
        #relation_definitions = self.define(input_list, extracted_triples)

        entities = {}
        triples = {}

        final_result_file = open(f"{output_dir}/kg.txt", "w")
        for idx, triplets in enumerate(extracted_triples):
            final_result_file.write(str(triplets))
            entities[idx] = set()
            triples[idx] = triplets

            for triplet in triplets:
                entities[idx].add(str(triplet[0]))
                entities[idx].add(str(triplet[2]))

            if idx != len(extracted_triples) - 1:
                final_result_file.write("\n")
            final_result_file.flush()
        
        return (entities, triples)


        