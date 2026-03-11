from argparse import ArgumentParser
from edc.edc_pipeline import EDC
from knowledge_graph.knowledge_graph import KnowledgeGraph
from models.hf_model import HFModel
from transformers import AutoTokenizer, AutoModelForCausalLM


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

    args = parser.parse_args()
    args_dict = vars(args)


    input_list = open(args_dict["input_file"], "r").readlines()

    hf_model = HFModel(args_dict["model"], device=args_dict["device"])


    extracted_text = hf_model.generate(input_list[0])
    print(extracted_text)
        


