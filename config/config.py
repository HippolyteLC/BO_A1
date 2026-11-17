from dataclasses import dataclass, field

from stage1.model.client import OllamaWrapper

import os

@dataclass
class ExperimentConfig:
    group_number: int = 12
    seeds: list[int] = field(default_factory=list)
    llamea_budget: int = 100
    dimension: int = 6
    initial_points: int = 10

    llamea_function_budget: int = 60 #10d
    experiment_evaluation_budget: int = 600 #100d

    upper: int = 1e3
    lower: int = 1e-2

    fall_back: int = 0

    task_prompt_name: str = "BO_STUB_TASK_PROMPT_1"

    def __post_init__(self):
        if not self.seeds:
            self.seeds = [10 * self.group_number + r for r in range(10)]

@dataclass
class ModelConfig:
    model_name: str = "qwen2.5-coder-14b-q4:latest"
    llm: OllamaWrapper = field(init=False, repr=False)
    temperature: float = 0.3
    context_window: int = 16384
    top_p: float = 0.95

    def __post_init__(self):
        os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1" 
        self.llm = OllamaWrapper(model=self.model_name, temperature=self.temperature, num_ctx=self.context_window, top_p=self.top_p)


# LOG_DIR = "results"

GROUP_NUM = 12

SEEDS = [10*GROUP_NUM+r for r in range(10)]

BO_GEN_BUDGET = 100

DIM = 6

INITIAL_POINTS = 10

FUNCTION_EVALS = 10*DIM