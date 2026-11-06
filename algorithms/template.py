"""Algorithm interface shared by all methods (generated algorithms and baselines).

Rules:
- __init__ receives budget, dim, initial_points, and random_seed (passed by the evaluator, seeds 120-129).
- Use random_seed for ALL randomness inside the algorithm; do not hard-code seeds.
- __call__ receives func; x must be a numpy array of shape (dim,).
- Search space is [-5, 5]^dim. Minimization.
- Evaluate only via func(x); initial design counts toward the budget.
- Return (best_f, best_x).
"""
import numpy as np


class RandomSearch:
    def __init__(self, budget, dim, random_seed):
        self.budget = budget
        self.dim = dim
        self.random_seed = random_seed

    def __call__(self, func):
        rng = np.random.default_rng(self.random_seed)
        best_f, best_x = np.inf, None
        for _ in range(self.budget):
            x = rng.uniform(-5, 5, size=self.dim)
            f = func(x)
            if f < best_f:
                best_f, best_x = f, x
        return best_f, best_x
