from bbob import BBOBFunction
from rw_problem import PracticalProblem

import numpy as np


def test_bbob_obj_f():
    obj = BBOBFunction(
        dimension=6,
        seed=42, 
        log_dir="test"
    )

    rng = np.random.default_rng(42)
    best = float("inf")
    try:
        for _ in range(1000):
            x = rng.uniform(-5, 5, size=obj.problem.meta_data.n_variables)
            best = min(best, obj.problem(x))
    finally:
        obj.problem.detach_logger()

    print(f"Best objective BBOB: {best}")

def test_practical_problem():
    obj = PracticalProblem(
        dimension=6,
        seed=42, 
        log_dir="test"
    )

    rng = np.random.default_rng(42)
    best = float("inf")
    try:
        for _ in range(1000):
            x = rng.uniform(-5, 5, size=obj.problem.meta_data.n_variables)
            best = min(best, obj.problem(x))
    finally:
        obj.problem.detach_logger()

    print(f"Best objective BBOB: {best}")

test_practical_problem()