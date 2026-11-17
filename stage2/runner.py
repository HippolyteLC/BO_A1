# Run with: python -m stage2.runner
# Stage 2: let every algorithm run on both problems, 10 seeds each, and save the results

import json
import platform
import random
import time
from pathlib import Path
 
import numpy as np
import torch
 
from algorithms.botorch_bo import BoTorchBO
from algorithms.cmaes import CMAES
from algorithms.template import RandomSearch
from benchmarks.bbob import BBOBFunction
from benchmarks.practical_problem import PracticalProblem
from config.config import ExperimentConfig

# change the prompt relative folder to select different llamea output algorithms. 
from algorithms.BO_STUB_TASK_PROMPT_2.real_Optimize_with_BO import Real_Optimize_with_BO
from algorithms.BO_STUB_TASK_PROMPT_2.bbob_Optimize_with_BO import BBOB_Optimize_with_BO

import datetime as dt

# -------------------  SETTINGS   ---------------------------------
 
ALGORITHMS = {
    "random_search": RandomSearch,
    "cmaes": CMAES,
    "botorch_bo": BoTorchBO,
    "llamea_real": Real_Optimize_with_BO, 
    "llamea_bbob": BBOB_Optimize_with_BO, 
}
 
ALGORITHMS_TO_RUN = ["random_search", "cmaes", "botorch_bo", "llamea_real", "llamea_bbob"]
PROBLEMS_TO_RUN = ["bbob", "real"]
 
config = ExperimentConfig()
BUDGET = config.experiment_evaluation_budget   # 600
# timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"results/stage2"                     
 
 
# -----------------  BUDGET COUNTER   ---------------------------------------
 
class BudgetExceeded(Exception):
    pass
 
 
class CountedFunction:
    # counts evaluations, stops at the budget
    # and remembers the best value found
 
    def __init__(self, problem, budget):
        self.problem = problem
        self.budget = budget
        self.n_evals = 0
        self.best = np.inf
        self.history = []
 
    def __call__(self, x):
        if self.n_evals >= self.budget:
            raise BudgetExceeded()
 
        x = np.array(x, dtype=float).flatten()
        y = float(self.problem(x))
        self.n_evals += 1
 
        if y < self.best:
            self.best = y
        self.history.append(self.best)
        return y
 
 

# --------------------   HELPERS   ----------------------------

 
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
 
 
def make_problem(name):
    
    if name == "bbob":
        problem = BBOBFunction(dimension=config.dimension, log_dir=None).problem
    elif name == "real":
        problem = PracticalProblem(dimension=config.dimension, log_dir=None).problem
    else:
        raise ValueError("Unknown problem: " + name)
 
    f_opt = problem.optimum.y   # best possible value: 1000.0 for f15, 0.0 for 1124
    return problem, f_opt
 
 
# --------------------------  ONE RUN  ------------------------------------------

 
def run_once(algorithm_class, problem_name, seed):
    problem, f_opt = make_problem(problem_name)   # a fresh problem for every run
    set_all_seeds(seed)
 
    func = CountedFunction(problem, BUDGET)
    algorithm = algorithm_class(
        budget=BUDGET,
        dim=config.dimension,
        initial_points=config.initial_points,
        random_seed=seed,
    )
 
    try:
        algorithm(func)
    except BudgetExceeded:
        pass   
 
    # If the algorithm stopped early, repeat last value
    curve = list(func.history)
    while len(curve) < BUDGET:
        curve.append(curve[-1])
 

    errors = []
    for value in curve:
        errors.append(value - f_opt)
 
    return curve, errors, func.n_evals, f_opt

# ---------------Smoke test--------------------------------------------------
# ------------------- SMOKE TEST --------------------------------

def smoke_test():
    print("\n========== SMOKE TEST ==========")

    test_budget = 10

    for algorithm_name, algorithm_class in ALGORITHMS.items():
        for problem_name in PROBLEMS_TO_RUN:
            print(f"Testing {algorithm_name} on {problem_name}...")

            problem, f_opt = make_problem(problem_name)
            set_all_seeds(42)

            func = CountedFunction(problem, test_budget)

            algorithm = algorithm_class(
                budget=test_budget,
                dim=config.dimension,
                initial_points=config.initial_points,
                random_seed=42,
            )

            try:
                algorithm(func)
            except BudgetExceeded:
                pass

            assert func.n_evals > 0, "No evaluations performed!"
            assert func.n_evals <= test_budget, "Budget exceeded!"
            assert np.isfinite(func.best), "Invalid objective value!"

            print(f"  OK | evaluations: {func.n_evals} | best: {func.best:.4f}")

    print("\n========== ALL TESTS PASSED ==========\n")

def diagnostic_smoke_test():
    print("\n========== DIAGNOSTIC SMOKE TEST ==========")

    test_budget = min(25, BUDGET)

    for algorithm_name, algorithm_class in ALGORITHMS.items():
        for problem_name in PROBLEMS_TO_RUN:
            print(f"Testing {algorithm_name} on {problem_name} (budget={test_budget})...")

            problem, f_opt = make_problem(problem_name)
            set_all_seeds(42)

            lb = np.asarray(problem.bounds.lb, dtype=float).reshape(-1)
            ub = np.asarray(problem.bounds.ub, dtype=float).reshape(-1)
            if lb.size == 1:
                lb = np.full(config.dimension, lb.item(), dtype=float)
            if ub.size == 1:
                ub = np.full(config.dimension, ub.item(), dtype=float)

            out_of_bounds_count = 0
            base_func = CountedFunction(problem, test_budget)

            def checked_func(x):
                nonlocal out_of_bounds_count
                arr = np.asarray(x, dtype=float).flatten()
                if np.any(arr < lb - 1e-9) or np.any(arr > ub + 1e-9):
                    out_of_bounds_count += 1
                return base_func(x)

            algorithm = algorithm_class(
                budget=test_budget,
                dim=config.dimension,
                initial_points=min(config.initial_points, max(1, test_budget // 3)),
                random_seed=42,
            )

            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                try:
                    algorithm(checked_func)
                except BudgetExceeded:
                    pass

            captured_out = buf.getvalue().strip()
            if captured_out:
                print(f"  [LOG] {captured_out}")

            assert base_func.n_evals > 0, "No evaluations performed!"
            assert base_func.n_evals <= test_budget, "Budget exceeded!"
            assert np.isfinite(base_func.best), "Invalid objective value!"
            assert out_of_bounds_count == 0, (
                f"Out-of-bounds evaluations detected ({out_of_bounds_count}) on {problem_name}!"
            )
            assert "failed" not in captured_out.lower(), (
                f"Fallback/error message printed during {algorithm_name} on {problem_name}: {captured_out}"
            )

            print(f"  OK | evals: {base_func.n_evals} | best: {base_func.best:.4f}")

    print("\n========== DIAGNOSTIC SMOKE TEST PASSED ==========\n")
 
# ---------------ALL SEEDS for one algorithm on one problem------------------

 
def run_all_seeds(algorithm_name, problem_name):
    algorithm_class = ALGORITHMS[algorithm_name]
 
    folder = Path(OUT_DIR) / algorithm_name / problem_name
    folder.mkdir(parents=True, exist_ok=True)
 
    for seed in config.seeds:
        path = folder / ("seed_" + str(seed) + ".json")
 
        if path.exists():
            print("skip (already done):", path)
            continue
 
        start = time.time()
        curve, errors, n_evals, f_opt = run_once(algorithm_class, problem_name, seed)
        seconds = time.time() - start
 
        result = {
            "algorithm": algorithm_name,
            "problem": problem_name,
            "seed": seed,
            "budget": BUDGET,
            "dimension": config.dimension,
            "initial_points": config.initial_points,
            "n_evals_used": n_evals,
            "f_opt": f_opt,
            "best_so_far": curve,          # best value after each evaluation
            "best_so_far_error": errors,   # the same, minus the optimum
            "final_error": errors[-1],
            "seconds": seconds,
            "machine": platform.platform(),
        }
        with open(path, "w") as f:
            json.dump(result, f)
 
        print("saved:", path, "| final error =", round(errors[-1], 4), "|", round(seconds, 1), "s")
 
 

# ----------   MAIN  ----------------

 
if __name__ == "__main__":
    smoke_test()
    diagnostic_smoke_test()
    # for algorithm_name in ALGORITHMS_TO_RUN:
    #     for problem_name in PROBLEMS_TO_RUN:
    #         print("===", algorithm_name, "on", problem_name, "| budget", BUDGET, "===")
    #         run_all_seeds(algorithm_name, problem_name)
