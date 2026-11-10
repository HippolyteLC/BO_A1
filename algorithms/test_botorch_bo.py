# Quick test of the BoTorch BO baseline. Run from the BO_A1 folder:
#     python -m algorithms.test_botorch_bo
#
# BO is slow, so this uses a small budget and only 3 seeds (takes a few minutes).
# Checks three things on both problems:
# 1. BO uses exactly the budget
# 2. it is better than random search
# 3. the same seed gives exactly the same result twice

import time

from algorithms.botorch_bo import BoTorchBO
from algorithms.template import RandomSearch
from stage2.runner import CountedFunction, make_problem, set_all_seeds

BUDGET = 30
SEEDS = [120, 121, 122]


def run(algorithm_class, problem_name, seed):
    problem, f_opt = make_problem(problem_name)
    set_all_seeds(seed)
    func = CountedFunction(problem, BUDGET)
    algorithm = algorithm_class(budget=BUDGET, dim=6, initial_points=10, random_seed=seed)
    algorithm(func)
    if func.n_evals != BUDGET:
        print("  PROBLEM: used", func.n_evals, "evaluations instead of", BUDGET)
    return func


for problem_name in ["bbob", "real"]:
    print("=== problem:", problem_name, "| budget", BUDGET, "===")

    for algorithm_class in [RandomSearch, BoTorchBO]:
        start = time.time()
        errors = []
        for seed in SEEDS:
            func = run(algorithm_class, problem_name, seed)
            problem, f_opt = make_problem(problem_name)
            errors.append(round(func.best - f_opt, 3))
        seconds = round((time.time() - start) / len(SEEDS), 1)
        print(" ", algorithm_class.__name__, "errors per seed =", errors, "|", seconds, "s per run")

    first = run(BoTorchBO, problem_name, 120).history
    second = run(BoTorchBO, problem_name, 120).history
    print("  same seed gives the same result:", first == second)
