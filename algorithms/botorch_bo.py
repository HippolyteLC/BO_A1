"""
Based on the sample LLaMEA candidate provided by the course
(https://github.com/BayesOptApp/BO_Course_Assignment, sample_LLaMEA_logs/code/try-0-Optimize_with_BO.py)

"""

import warnings

import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc
from torch.quasirandom import SobolEngine


 
class BoTorchBO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = min(initial_points, budget)
        self.random_seed = random_seed
        self.lb = -5.0   # lower bound
        self.ub = 5.0    # upper bound
 
    # ----- initial design -----
 
    def initial_design(self):
        # spreads the points evenly over the search space
        # sampler = qmc.LatinHypercube(d=self.dim, seed=self.random_seed)
        # points_01 = sampler.random(n=self.initial_points)            # points in [0, 1]^dim
        # # return points_01 * (self.ub - self.lb) + self.lb             # scaled to [-5, 5]^dim

        sobol_engine = SobolEngine(dimension=self.dim, scramble=True, seed=self.random_seed)
        X_init = sobol_engine.draw(self.initial_points).numpy()
        X_scaled = self.lb + X_init * (self.ub - self.lb)
        return X_scaled
    # ----- surrogate model -----
 
    def fit_gp(self, X, y):
        # Fit a Gaussian Process to all points evaluated so far
        X_tensor = torch.tensor(np.array(X), dtype=torch.double)
        y_tensor = torch.tensor(y, dtype=torch.double).unsqueeze(-1)  # shape (n, 1)
        minus_y = -y_tensor   # BoTorch maximizes, so we model -f to minimize f
 
        model = SingleTaskGP(
            X_tensor,
            minus_y,
            input_transform=Normalize(d=self.dim, bounds=self.bounds),   # inputs to [0, 1]
            outcome_transform=Standardize(m=1),                          # outputs to mean 0, std 1
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)   # learn the GP hyperparameters
        return model, minus_y.max()
 
    # ----- acquisition function -----
 
    def propose_next_point(self, model, best_minus_y):
        # Find the point with the highest log Expected Improvement
        acquisition = LogExpectedImprovement(model=model, best_f=best_minus_y)
        candidate, _ = optimize_acqf(
            acq_function=acquisition,
            bounds=self.bounds,
            q=1,               # one new point at a time
            num_restarts=10,   # start the optimizer from 10 different points
            raw_samples=256,   # random points used to choose those 10 starting points
        )
        return candidate.detach().squeeze(0).numpy()
 
    # ----- main -----
 
    def __call__(self, func):
        torch.manual_seed(self.random_seed)
        rng = np.random.default_rng(self.random_seed)
        self.bounds = torch.tensor([[self.lb] * self.dim, [self.ub] * self.dim], dtype=torch.double)
 
        # Initial design
        X = []
        y = []
        for x in self.initial_design():
            X.append(x)
            y.append(func(x))
 
        # BO loop: fit the GP, pick the next point, evaluate, repeat
        while len(y) < self.budget:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")   # BoTorch prints many harmless warnings
                    model, best_minus_y = self.fit_gp(X, y)
                    x_next = self.propose_next_point(model, best_minus_y)
            except Exception:
                x_next = rng.uniform(self.lb, self.ub, size=self.dim)   # rare fallback
 
            X.append(x_next)
            y.append(func(x_next))
 
        best_index = int(np.argmin(y))
        return y[best_index], np.array(X[best_index])
