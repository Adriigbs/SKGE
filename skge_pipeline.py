from argparse import ArgumentParser
from models.hf_model import HFModel
from edc.edc_pipeline import EDC 
from planbench.prompt_parser import PromptParser
from config_parser import EdcConfig, PipelineConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Planner:

    def __init__(self, model, config, problem_prompt, problem_type="blocksworld"):
        self.config = config
        self.history = {}
        self.model_wrapper = HFModel(model)
        self.problem_prompt = problem_prompt
        self.plan_bench_prompt_parser = PromptParser(problem_type)


    def ask(self):

        system_prompt, user_prompt = self.plan_bench_prompt_parser.parse(self.problem_prompt)

        self.prepare_system_prompt(system_prompt)

        for step in range(self.config.max_steps):
            print("Yo")
            step_output = self.model_wrapper.generate(self.system_prompt, user_prompt)
            logger.info(f"Step {step+1} output: {step_output}")
            



    def prepare_system_prompt(self, problem_prompt):
        
        base = """You are a reasoning agent. At each step propose exactly one reasoning step.
                Wait for verification before proceeding.
                Do not explain. Do not plan ahead. Do not propose multiple steps.
                """
                
        self.system_prompt = base + problem_prompt
        





if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--se_template", type=str)
    parser.add_argument("--se_fewshot", type=str)
    parser.add_argument("--sd_template", type=str)
    parser.add_argument("--sd_fewshot", type=str)
    parser.add_argument("--max_steps", type=int, default=5)

    args_dict = vars(parser.parse_args())

    pipeline_config = PipelineConfig(
        max_steps=args_dict["max_steps"]
    )

    
    pipeline = Planner(
        model=args_dict["model"],
        config=pipeline_config,
        problem_prompt=open(args_dict["input_file"]).read(),
    )

    pipeline.ask()