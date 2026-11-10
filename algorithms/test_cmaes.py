# Quick test of the CMA-ES baseline. Run from the BO_A1 folder:
#     python -m algorithms.test_cmaes
#
# Checks three things on both problems:
# 1. CMA-ES uses exactly the budget (not more, not less)
# 2. it is better than random search
# 3. the same seed gives exactly the same result twice

from algorithms.cmaes import CMAES
from algorithms.template import RandomSearch
from stage2.runner import CountedFunction, make_problem

BUDGET = 100

for problem_name in ["bbob", "real"]:
    print("=== problem:", problem_name, "| budget", BUDGET, "===")

    for algorithm_class in [RandomSearch, CMAES]:
        errors = []
        for seed in range(120, 130):
            problem, f_opt = make_problem(problem_name)
            func = CountedFunction(problem, BUDGET)
            algorithm = algorithm_class(budget=BUDGET, dim=6, initial_points=10, random_seed=seed)
            algorithm(func)

            if func.n_evals != BUDGET:
                print("  PROBLEM: used", func.n_evals, "evaluations instead of", BUDGET)
            errors.append(func.best - f_opt)

        errors.sort()
        median_error = (errors[4] + errors[5]) / 2
        print(" ", algorithm_class.__name__, "median error over 10 seeds =", round(median_error, 3))

    # Reproducibility: run the same seed twice
    histories = []
    for attempt in range(2):
        problem, f_opt = make_problem(problem_name)
        func = CountedFunction(problem, BUDGET)
        CMAES(budget=BUDGET, dim=6, initial_points=10, random_seed=120)(func)
        histories.append(func.history)
    print("  same seed gives the same result:", histories[0] == histories[1])
