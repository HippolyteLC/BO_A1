from llamea import LLaMEA, Ollama_LLM

import os

from benchmarks.practical_problem import PracticalProblem

import re
import textwrap
import math
import numpy as np

from llamea. import aoc_logger



# We implement an exception to stop algorithms that try a too large budget (to prevent infinite loops).
class OverBudgetException(Exception):
    """The algorithm tried to do more evaluations than allowed."""
    pass

LOG_DIR = "results"

GROUP_NUM = 12

SEEDS = [10*GROUP_NUM+r for r in range(10)]

BO_GEN_BUDGET = 100

DIM = 6

FUNCTION_EVALS = 10*DIM

ROLE_PROMPT = """\
You are an AI that generates Python optimization code by using Bayesian Optimization.
"""
TASK_PROMPT = """\
Create a Python class named Optimize_with_BO, with __init__(self, budget, dim, initial_points, random_seed) that initializes the budget and dimension, the initial DoE points and the random seed (for ulterior Sobol samplers from torch or for LHS from scipy),
and a __call__(self, func) method that tries to find x in [-5, 5]^dim that minimizes func(x) (this can be set randomly, but imagine it's a highly multimodal function like the Gallagher 101 peaks).
The function should return the tuple best_f, best_x (value best found and the location).
Implement an efficient algorithm for this Bayesian Optimization task. The code you implement must
only use BoTorch and GPyTorch libraries for the optimization; 
You may also use scipy to generate random samples for the initial DoE and then transform the variables fromlists or numpy arrays into torch tensors. 
Do not use any other optimization libraries. You may append additional helper functions or classes if needed, but the main class must be named Optimize_with_BO and must have the specified methods.
You can also use other BoTorch acquisition functions like qNoisyExpectedImprovement, qLogNoisyExpectedImprovement, qExpectedImprovement, qLogExpectedImprovement, qUpperConfidenceBound, and you can use the MaternKernel and ScaleKernel from GPyTorch.
You can also design your own acquisition function if you want, but it must be based on the BoTorch framework. Check for example the SAWEI from Benjamins et al. (2023) paper, which is a combination of qLogExpectedImprovement and qUpperConfidenceBound.
Or for instance take a look at the paper from HEBO and implement a GP-Hedge balancing the acquisition functions.

An example of such code (a simple random search), is as follows:
```
import numpy as np

class RandomSearch:
    def __init__(self, budget: int, dim: int, initial_points: int, random_seed: int):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed(self.random_seed)
        for i in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)

            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

        return self.f_opt, self.x_opt
```

Provide the Python code and a one-line description with the main idea (without enters). Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```
"""

# cite the example from llamea GH

def evaluate_simple(solution, explogger=None):
    code_str = solution.code  # The Python code the LLM generated
    alg_name = solution.name

    # We define our 6D search space: x in [-5, 5]^6, budget=60 func evals
    # We'll create a small function that the generated code should optimize.

    problem = PracticalProblem(dimension=DIM, seed=42, log_dir="test_log")
    def f(x):
        # We only allow so many function calls
        if f.call_count >= FUNCTION_EVALS:
            raise OverBudgetException("Budget exceeded.")
        f.call_count += 1
        return problem(x)
    f.call_count = 0

    # Dynamically run the generated code
    # The code is expected to define a class named alg_name, with __init__(budget, dim) and __call__(f).
    # We'll create a safe execution context to run it.
    safe_globals = {
        "OverBudgetException": OverBudgetException,
        "math": math,
        "np": np,
    }
    local_env = {}
    try:
        exec(code_str, safe_globals, local_env)
    except Exception as e:
        # If there's an error in code, set the score to 0
        solution.set_scores(0, feedback=f"Runtime/Syntax error: {e}")
        return solution


    # Now run the algorithm
    best_f = math.inf
    try:
        best_f, best_x = algo(f)
    except OverBudgetException:
        # If over budget, we penalize heavily
        best_f = 9999

    # test algorithm init 
    try:
        AlgorithmClass = local_env[alg_name]
        algo = AlgorithmClass(budget=FUNCTION_EVALS, dim=DIM, initial_points=10)
    except Exception as e:
        solution.set_scores(0, feedback=f"Instantiation error: {e}")
        return solution

    problem = PracticalProblem(dimension=DIM, log_dir=LOG_DIR)
    for seed in SEEDS:
        np.random.seed(seed)
        try:
            algo = AlgorithmClass(budget=FUNCTION_EVALS, dim=DIM, initial_points=10, random_seed=seed)
        except OverBudgetException:
            pass
        auc = correct_aoc(problem, l2, budget)
        aucs.append(auc)
        l2.reset(problem)
        problem.reset()


    # We'll convert it to a "score" where smaller f is better => we do `score = 1/(1 + best_f)`
    # so that 0 => 1/1 => 1, big f => near 0
    # Note: LLaMEA is optimizing by default! (bigger is better)
    score = 1.0 / (1.0 + best_f)

    # Provide feedback
    feedback_str = f"Algorithm {alg_name} got score={score:.4f} (bigger is better)."

    # Save the score to the solution object
    solution.set_scores(score, feedback_str)
    return solution

if __name__=="__main__":
    AI_MODEL = "Llama-3.2-3B-Instruct-Q4_K_M:latest"                     
    os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1" 
    llm = Ollama_LLM(model=AI_MODEL)

    role_prompt = textwrap.dedent(ROLE_PROMPT)
    task_prompt = textwrap.dedent(TASK_PROMPT)

    # We'll use a small number of iterations for demonstration
    es = LLaMEA(
        f=evaluate_simple,
        llm=llm,
        n_parents=1,
        n_offspring=1,
        role_prompt=role_prompt,
        task_prompt=task_prompt,
        experiment_name="bo-gen-test-1",
        elitism=True, # this promotes local optima (1+1)
        budget=BO_GEN_BUDGET  # Try 3 iterations for a quick demo
    )

    best_solution = es.run()
    print(f"Best found solution: {best_solution.name}, Score={best_solution.fitness:.4f}")
    print(f"Generated code:\n{best_solution.code}")
    print(f"Additional feedback: {best_solution.feedback}")