import os

import ioh
import numpy as np
from Practical_Problems import get_practical_problem

problem = get_practical_problem(1124, dim=6)

logger = ioh.logger.Analyzer(
    triggers=[ioh.logger.trigger.OnImprovement(), ioh.logger.trigger.Each(10)],
    root=os.path.curdir,
    algorithm_name="RandomSearch",
    folder_name="Results-Practical",
    algorithm_info="Uniform random search",
    store_positions=False,
)
problem.attach_logger(logger)

rng = np.random.default_rng(42)
best = float("inf")
try:
    for _ in range(1000):
        x = rng.uniform(-5, 5, size=problem.meta_data.n_variables)
        best = min(best, problem(x))
finally:
    problem.detach_logger()

print(f"Best objective: {best}")