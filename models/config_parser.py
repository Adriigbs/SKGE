from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str
    max_length: int
    max_new_tokens: int