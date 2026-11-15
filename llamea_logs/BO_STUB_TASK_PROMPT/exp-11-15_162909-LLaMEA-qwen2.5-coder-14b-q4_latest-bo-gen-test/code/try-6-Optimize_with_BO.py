import numpy as np
import torch
from torch.quasirandom import SobolEngine
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel

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
        train_X = X.double()
        train_Y = y.unsqueeze(-1).double()
        covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
        model = SingleTaskGP(train_X, train_Y, covar_module=covar_module, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model, -torch.max(train_Y)

    def _next_point(self, model, best_f_max, X: torch.Tensor, y: torch.Tensor, step: int) -> torch.Tensor:
        acq_func = ExpectedImprovement(model, best_f_max)
        candidates, _ = optimize_acqf(
            acq_func, bounds=self.unit_bounds, q=1, num_restarts=5, raw_samples=64, options={"maxiter": 200}
        )
        return candidates.detach()

    def __call__(self, func):
        X = self._initial_design(self.n_init)
        lb, ub = func.bounds.lb, func.bounds.ub
        X_scaled = X * (ub - lb) + lb
        y = torch.tensor([func(x.numpy()) for x in X_scaled], dtype=torch.double)
        model, best_f_max = self._fit_model(X, y)
        best_f = float(y.min())
        best_x = X_scaled[y.argmin()].numpy()

        for step in range(self.n_init, self.budget):
            next_x = self._next_point(model, best_f_max, X, y, step)
            next_x_scaled = next_x * (ub - lb) + lb
            next_y = func(next_x_scaled.numpy())
            if next_y < best_f:
                best_f = next_y
                best_x = next_x_scaled.numpy()
            X = torch.cat([X, next_x])
            y = torch.cat([y, torch.tensor([next_y], dtype=torch.double)])
            model, best_f_max = self._fit_model(X, y)

        return best_f, best_x