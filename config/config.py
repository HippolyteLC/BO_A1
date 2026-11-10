from dataclasses import dataclass, field

from llamea import Ollama_LLM

@dataclass
class ExperimentConfig:
    group_number: int = 12
    seeds: list[int] = field(default_factory=list)
    llamea_budget: int = 100
    dimension: int = 6
    initial_points: int = 10

    llamea_function_budget: int = 60 #10d
    experiment_evaluation_budget: int = 600 #100d

    def __post_init__(self):
        if not self.seeds:
            self.seeds = [10 * self.group_number + r for r in range(10)]

@dataclass
class ModelConfig:
    model_name: str = "qwen2.5-coder-14b-q4:latest"
    llm: Ollama_LLM = field(init=False, repr=False)

    def __post_init__(self):
        self.llm = Ollama_LLM(model=self.model_name)


# LOG_DIR = "results"

GROUP_NUM = 12

SEEDS = [10*GROUP_NUM+r for r in range(10)]

BO_GEN_BUDGET = 100

DIM = 6

INITIAL_POINTS = 10

FUNCTION_EVALS = 10*DIM