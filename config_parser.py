from dataclasses import dataclass



@dataclass
class EdcConfig:
    se_prompt: str
    se_fewshot: str
    sd_prompt: str
    sd_fewshot: str


@dataclass
class PipelineConfig:
    max_steps: int