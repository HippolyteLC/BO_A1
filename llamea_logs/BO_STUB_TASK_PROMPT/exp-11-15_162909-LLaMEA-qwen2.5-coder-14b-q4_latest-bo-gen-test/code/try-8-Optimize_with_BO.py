import numpy as np
import torch
from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int, random_seed: int):
        self.budget = budget
        self.dim = dim
        self.n_init = min(initial_points, budget)
        self.seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.sobol = SobolEngine(dimension=dim, scramble=True, seed=random_seed)
        self.unit_bounds = torch.stack([torch.zeros(dim), torch.ones(dim)]).double()

    def _initial_design(self, n: int) -> torch.Tensor:
        return self.sobol.draw(n).double()

    def _fit_model(self, X: torch.Tensor, y: torch.Tensor):
        X_standardized = standardize(X)
        model = SingleTaskGP(X_standardized, -y.view(-1, 1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model, -y.max().item()

    def _next_point(self, model, best_f_max, X: torch.Tensor, y: torch.Tensor, step: int) -> torch.Tensor:
        acquisition = ExpectedImprovement(model, best_f_max)
        candidates, _ = optimize_acqf(
            acquisition, bounds=self.unit_bounds, q=1, num_restarts=5, raw_samples=64, return_best_only=True
        )
        return candidates

    def __call__(self, func):
        lb, ub = -5, 5
        X = self._initial_design(self.n_init)
        X_scaled = unnormalize(X, torch.tensor([lb]*self.dim), torch.tensor([ub]*self.dim))
        y = torch.tensor([func(x.numpy()) for x in X_scaled])
        
        best_f = y.min().item()
        best_x = X_scaled[y.argmin()].numpy()

        for step in range(self.n_init, self.budget):
            model, best_f_max = self._fit_model(X, y)
            next_x = self._next_point(model, best_f_max, X, y, step)
            next_x_scaled = unnormalize(next_x, torch.tensor([lb]*self.dim), torch.tensor([ub]*self.dim))
            next_y = func(next_x_scaled.numpy())
            
            if next_y < best_f:
                best_f = next_y
                best_x = next_x_scaled.numpy()
            
            X = torch.cat([X, next_x])
            y = torch.cat([y, torch.tensor([next_y])])
        
        return best_f, best_x