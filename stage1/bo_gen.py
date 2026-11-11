from llamea import LLaMEA

from benchmarks.practical_problem import PracticalProblem
from benchmarks.bbob import BBOBFunction

from stage1.utils import silence_warnings

from ioh import logger, get_problem

from config.config import ExperimentConfig, ModelConfig

from stage1.prompts import VANILLA_ROLE_PROMPT, VANILLA_TASK_PROMPT, ENGINEERED_ROLE_PROMPT, ENGINEERED_TASK_PROMPT, OPEN_TASK_PROMPT, BO_TEMPLATE_TASK_PROMPT

import re
import textwrap
import os
import math
import numpy as np

from stage1.evaluator import Evaluator

# Silence warnings for cleaner terminal logs
silence_warnings()

config = ExperimentConfig()

PROBLEM = "real"

if __name__=="__main__":
    role_prompt = textwrap.dedent(ENGINEERED_ROLE_PROMPT)
    task_prompt = textwrap.dedent(OPEN_TASK_PROMPT)

    evaluator = Evaluator(config, PROBLEM)

    es = LLaMEA(
        f=evaluator,
        llm=ModelConfig().llm,
        n_parents=2,
        n_offspring=4,
        role_prompt=role_prompt,
        task_prompt=BO_TEMPLATE_TASK_PROMPT,
        experiment_name="bo-gen-test",
        elitism=True, # this promotes local optima (1+1)
        budget=config.llamea_budget  
    )

    best_solution = es.run()
    print(f"Best found solution: {best_solution.name}, Score={best_solution.fitness:.4f}")
    print(f"Generated code:\n{best_solution.code}")
    print(f"Additional feedback: {best_solution.feedback}")


