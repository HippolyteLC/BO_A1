# Skeleton code examples for the llamea prompt: BO_SKELETON_TASK_PROMPT
# Can paste any skeleton example used in a prompt into this file 
# and run test.py to verify if the skeleton works. 

import numpy as np
import torch
from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int, random_seed: int):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed

    def _evaluate(self, func, x_unit, lb, ub):
        x = lb + (ub - lb) * x_unit
        return float(func(x.detach().cpu().numpy()))

    def __call__(self, func):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        lb = torch.as_tensor(np.asarray(func.bounds.lb, dtype=float).reshape(-1), dtype=torch.double)
        ub = torch.as_tensor(np.asarray(func.bounds.ub, dtype=float).reshape(-1), dtype=torch.double)
        unit_bounds = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)]).to(torch.double)

        n_init = min(self.initial_points, self.budget)
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.random_seed)
        X = sobol.draw(n_init).to(torch.double)
        Y = torch.tensor([self._evaluate(func, x, lb, ub) for x in X], dtype=torch.double)

        while Y.shape[0] < self.budget:
            z = torch.log10(1.0 + Y.clamp_min(0.0))
            z = (z - z.mean()) / (z.std() + 1e-9)
            train_Y = (-z).unsqueeze(-1)  # shape (n, 1), larger is better

            model = SingleTaskGP(X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            acq = qLogExpectedImprovement(model=model, best_f=train_Y.max())
            cand, _ = optimize_acqf(acq, bounds=unit_bounds, q=1, num_restarts=5, raw_samples=64)
            x_new = cand.detach().reshape(1, -1)

            y_new = self._evaluate(func, x_new.squeeze(0), lb, ub)
            X = torch.cat([X, x_new], dim=0)
            Y = torch.cat([Y, torch.tensor([y_new], dtype=torch.double)])

        i = int(torch.argmin(Y))
        best_x = (lb + (ub - lb) * X[i]).numpy()
        return float(Y[i]), best_x