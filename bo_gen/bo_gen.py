from llamea import LLaMEA

import os

from benchmarks.practical_problem import PracticalProblem
from benchmarks.bbob import BBOBFunction

from misc.utils import aoc_logger, correct_aoc, OverBudgetException

from ioh import logger, get_problem

from config.config import ExperimentConfig, ModelConfig

from prompts import VANILLA_ROLE_PROMPT, VANILLA_TASK_PROMPT, ENGINEERED_ROLE_PROMPT, ENGINEERED_TASK_PROMPT

import re
import textwrap
import math
import numpy as np

from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# cite the example from llamea GH

config = ExperimentConfig()

# TODO: ensure the problem is made a minimization task

def evaluateBBOB(solution, explogger=None):
    auc_mean = 0
    auc_std = 0

    code = solution.code
    algorithm_name = solution.name

    try:
        exec(code, globals())
    except Exception as e:
        # If there's an error in code, set the score to 0
        solution.set_scores(0, feedback=f"Runtime/Syntax error: {e}")
        print("DEFAULTED TO THIS")
        return solution

    # exec(code, globals())

    # TODO: add keepeing track of budget

    error = ""

    aucs = []

    algorithm = None

    budget = config.llamea_function_budget
    dim = config.dimension

    l2 = aoc_logger(budget=budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
    problem = PracticalProblem(dimension=dim).problem
    problem.attach_logger(l2)

    for seed in config.seeds:
        np.random.seed(seed)
        try:
            algorithm = globals()[algorithm_name](
                budget=budget, dim=dim, initial_points=config.initial_points, random_seed=seed
            )
            algorithm(problem)
        except OverBudgetException:
            pass
        except Exception as e:
            print("We are failing RIGHT HERE...")
            error_feedback = (
                f"Your generated algorithm crashed during execution.\n"
                f"Exception: {type(e).__name__}\n"
                f"Details: {str(e)}\n"
                f"Please correct the implementation."
            )
            print(error_feedback)
            solution.add_metadata("error", error_feedback)
            solution.set_scores(0.0, error_feedback)
            return solution

        auc = correct_aoc(problem, l2, budget)
        aucs.append(auc)
        l2.reset(problem)
        problem.reset()
    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)

    feedback = f"The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.2f} with standard deviation {auc_std:0.2f}."

    print(algorithm_name, algorithm, auc_mean, auc_std)
    solution.add_metadata("aucs", aucs)
    solution.set_scores(auc_mean, feedback)

    return solution

if __name__=="__main__":
    # AI_MODEL = "Llama-3.2-3B-Instruct-Q4_K_M:latest" 
    llm = ModelConfig().llm
    os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1" 

    

    role_prompt = textwrap.dedent(ENGINEERED_ROLE_PROMPT)
    task_prompt = textwrap.dedent(ENGINEERED_TASK_PROMPT)

    # We'll use a small number of iterations for demonstration
    es = LLaMEA(
        f=evaluateBBOB,
        llm=llm,
        n_parents=1,
        n_offspring=1,
        role_prompt=role_prompt,
        task_prompt=task_prompt,
        experiment_name="logsbo-gen-test",
        elitism=False, # this promotes local optima (1+1)
        budget=config.llamea_budget  
    )

    best_solution = es.run()
    print(f"Best found solution: {best_solution.name}, Score={best_solution.fitness:.4f}")
    print(f"Generated code:\n{best_solution.code}")
    print(f"Additional feedback: {best_solution.feedback}")

