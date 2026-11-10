
import random
from pathlib import Path

import ioh
import numpy as np
import torch
import yaml
from Practical_Problems import get_practical_problem

ROOT = Path(__file__).resolve().parent


def load_config(path=ROOT / "config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def make_bbob_problem(cfg):
    # read from config
    function_id = cfg["problems"]["bbob"]["function_id"]   # 15
    instance = cfg["problems"]["bbob"]["instance"]         # 1
    dim = cfg["dimension"]                                 # 6

    # create BBOB
    problem = ioh.get_problem(function_id, instance=instance, dimension=dim,
                              problem_class=ioh.ProblemClass.BBOB)

    # read optimum value (BBOB)
    f_opt = problem.optimum.y
    return problem, f_opt


def make_real_problem(cfg):
    problem_id = cfg["problems"]["real"]["problem_id"]   # 1124
    dim = cfg["dimension"]                               # 6

    problem = get_practical_problem(problem_id, dim=dim)

    # read optimum value (real/1124)
    f_opt = cfg["problems"]["real"]["optimum"]           # 0.0
    return problem, f_opt


def make_problem(name, cfg):
    if name == "bbob":
        return make_bbob_problem(cfg)
    if name == "real":
        return make_real_problem(cfg)
    raise ValueError("Unknown problem: " + name)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BudgetExceeded(Exception):
    pass


class CountedFunction:
    # enforces the budget and records the best-so-far curve

    def __init__(self, problem, budget):
        self.problem = problem
        self.budget = budget
        self.n_evals = 0
        self.best = np.inf # best value
        self.history = []  # best-so-far after each evaluation

    def __call__(self, x):

        if self.n_evals >= self.budget: # check budget
            raise BudgetExceeded()

        x = np.array(x, dtype=float).flatten()

        y = float(self.problem(x))
        self.n_evals += 1

        self.best = min(self.best, y)
        self.history.append(self.best)

        return y
