import numpy as np
import torch
from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel

class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int, random_seed: int):
        self.budget = budget
        self.dim = dim
        self.random_seed = random_seed
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X = None
        self.y = None
        self.n_evals = 0
        self.n_init = min(initial_points, budget)
        self.trust_region_radius = 0.5
        self.trust_region_decay = 0.95

    def _sample_points(self, n_points):
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.random_seed)
        X = sobol.draw(n_points).numpy()
        X = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * X
        return X

    def _fit_model(self, X, y):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        X_tensor = torch.tensor(X, dtype=torch.double)
        y_tensor = torch.tensor(y, dtype=torch.double).unsqueeze(-1)
        kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
        model = SingleTaskGP(X_tensor, y_tensor, covar_module=kernel)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, options={"maxiter": 100})
        return model

    def _acquisition_function(self, model, X):
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        acqf = qLogExpectedImprovement(model, torch.tensor(self.y.max(), dtype=torch.double), num_fantasies=16)
        return acqf(X).detach().cpu().numpy()

    def _select_next_points(self, model):
        if self.n_evals < self.n_init:
            return self._sample_points(1)

        incumbent = self.X[self.y.argmin()]
        trust_region = self.trust_region_radius
        X_cand = self._sample_points(100)
        X_cand = np.clip(X_cand, incumbent - trust_region, incumbent + trust_region)
        X_cand = np.unique(X_cand, axis=0)

        X_tensor = torch.tensor(X_cand, dtype=torch.double)
        acqf_values = self._acquisition_function(model, X_tensor)
        best_idx = np.argmax(acqf_values)
        X_next = X_cand[best_idx:best_idx + 1]

        self.trust_region_radius *= self.trust_region_decay
        return X_next

    def _evaluate_points(self, func, X):
        max_evals = self.budget - self.n_evals
        X = X[:max_evals]
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(self, func):
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        while self.n_evals < self.budget:
            model = self._fit_model(self.X, self.y)
            X_next = self._select_next_points(model)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

        best_idx = self.y.argmin()
        best_f = self.y[best_idx, 0]
        best_x = self.X[best_idx]
        return best_f, best_x