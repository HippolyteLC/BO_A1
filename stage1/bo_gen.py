# To run: python -m stage1.bo_gen

from llamea import LLaMEA

from stage1.utils import silence_warnings

from config.config import ExperimentConfig, ModelConfig

from stage1.prompts.exp_prompts import *

import textwrap

from stage1.evaluator import Evaluator

from pathlib import Path


# Silence warnings for cleaner terminal logs
silence_warnings()

config = ExperimentConfig()


if __name__=="__main__":
    role_prompt = textwrap.dedent(ENGINEERED_ROLE_PROMPT)
    task_prompt_name = "BO_STUB_TASK_PROMPT_1"
    task_prompt = textwrap.dedent(globals()[task_prompt_name])

    for problem in ["real", "bbob"]:
        evaluator = Evaluator(config, problem, testing=True)
        
        es = LLaMEA(
            f=evaluator,
            llm=ModelConfig().llm,
            n_parents=2,
            n_offspring=5,
            role_prompt=role_prompt,
            task_prompt=task_prompt,
            experiment_name=f"{problem.capitalize()}",
            elitism=True, # this promotes local optima (2+5)
            budget=config.llamea_budget,
        )

        best_solution = es.run()
        print(f"Best found solution: {best_solution.name}, Score={best_solution.fitness:.4f}")
        # print(f"Generated code:\n{best_solution.code}")
        print(f"Additional feedback: {best_solution.feedback}")

        # Save the best-found algorithm only if its code is valid
        output_dir = Path("algorithms") / task_prompt_name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{problem}_Optimize_with_BO.py"

        try:
            compile(best_solution.code, str(output_file), "exec")

            output_file.write_text(best_solution.code, encoding="utf-8")
            print(f"Best algorithm saved to: {output_file}")

        except SyntaxError as e:
            print(f"Invalid Python code. Algorithm not saved.")
            print(f"Syntax error: {e}")


