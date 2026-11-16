# To run: python -m stage1.bo_gen

from llamea import LLaMEA

from stage1.utils import silence_warnings

from config.config import ExperimentConfig, ModelConfig

from stage1.prompts.exp_prompts import *

import textwrap

from stage1.evaluator import Evaluator

# Silence warnings for cleaner terminal logs
silence_warnings()

config = ExperimentConfig()


if __name__=="__main__":
    role_prompt = textwrap.dedent(ENGINEERED_ROLE_PROMPT)
    task_prompt = textwrap.dedent(BO_STUB_TASK_PROMPT_2)

    for problem in ["real", "bbob"]:
        evaluator = Evaluator(config, problem, testing=False)
        
        es = LLaMEA(
            f=evaluator,
            llm=ModelConfig().llm,
            n_parents=2,
            n_offspring=5,
            role_prompt=role_prompt,
            task_prompt=task_prompt,
            experiment_name=f"{problem.capitalize()}",
            elitism=True, # this promotes local optima (1+1)
            budget=config.llamea_budget,
        )

        best_solution = es.run()
        print(f"Best found solution: {best_solution.name}, Score={best_solution.fitness:.4f}")
        # print(f"Generated code:\n{best_solution.code}")
        print(f"Additional feedback: {best_solution.feedback}")


