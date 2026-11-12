import numpy as np
import torch
from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood

class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int, random_seed: int):
        self.budget = budget
        self.dim = dim
        self.n_init = min(initial_points, budget // 2)  # Smaller initial design
        self.seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.sobol = SobolEngine(dimension=dim, scramble=True, seed=random_seed)
        self.unit_bounds = torch.stack([torch.zeros(dim), torch.ones(dim)]).double()

    def _initial_design(self, n):
        return self.sobol.draw(n).double()

    def _fit_model(self, X, y):
        z = torch.log10(1.0 + y.clamp_min(0.0))
        z = (z - z.mean()) / (z.std() + 1e-9)
        train_Y = (-z).unsqueeze(-1)
        model = SingleTaskGP(X, train_Y)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        return model, train_Y.max()

    def _next_point(self, model, best_f, X, y, iteration):
        if iteration < self.budget * 0.5:
            acq = qLogExpectedImprovement(model, best_f=best_f)
        else:
            acq = qUpperConfidenceBound(model, beta=0.2)
        xb = X[torch.argmin(y)]
        r = 0.5 / (1 + np.exp(-0.05 * (iteration - self.n_init)))  # Adjusted decay rate for trust region radius
        bounds = torch.stack([(xb - r).clamp(0, 1), (xb + r).clamp(0, 1)])
        cand, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=5, raw_samples=64)
        return cand.detach()

    def _evaluate(self, func, x_unit, lb, ub):
        x = lb + (ub - lb) * x_unit.detach().cpu().numpy()
        return float(func(x))

    def __call__(self, func):
        lb = np.asarray(func.bounds.lb, dtype=float).reshape(-1)
        ub = np.asarray(func.bounds.ub, dtype=float).reshape(-1)
        X = self._initial_design(self.n_init)
        y = torch.tensor([self._evaluate(func, x, lb, ub) for x in X], dtype=torch.double)
        for iteration in range(self.n_init, self.budget):
            model, best_f = self._fit_model(X, y)
            x_new = self._next_point(model, best_f, X, y, iteration)
            y_new = self._evaluate(func, x_new[0], lb, ub)
            X = torch.cat([X, x_new], dim=0)
            y = torch.cat([y, torch.tensor([y_new], dtype=torch.double)])
        i = int(torch.argmin(y))
        return float(y[i]), lb + (ub - lb) * X[i].numpy()