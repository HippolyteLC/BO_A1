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
        self.trust_radius = 0.5  # Initial trust region radius

    def _sample_points(self, n_points):
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.random_seed)
        samples = sobol.draw(n_points).detach().cpu().numpy()
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * samples

    def _fit_model(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.double)
        y_tensor = torch.tensor(y, dtype=torch.double).unsqueeze(-1)
        kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim))
        model = SingleTaskGP(X_tensor, y_tensor)
        model.covar_module = kernel
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, options={"maxiter": 100})
        return model

    def _acquisition_function(self, X):
        model = self._fit_model(self.X, self.y)
        acqf = qLogExpectedImprovement(model, torch.tensor(self.y.max(), dtype=torch.double).unsqueeze(-1))
        return acqf(torch.tensor(X, dtype=torch.double)).detach().cpu().numpy()

    def _select_next_points(self, batch_size):
        if self.n_evals < self.n_init:
            return self._sample_points(batch_size)
        
        incumbent = self.X[self.y.argmin()]
        trust_region = np.array([incumbent - self.trust_radius, incumbent + self.trust_radius])
        trust_region = np.clip(trust_region, self.bounds[0], self.bounds[1])
        
        candidates = self._sample_points(100)
        candidates = candidates[(candidates >= trust_region[0]).all(axis=1) & (candidates <= trust_region[1]).all(axis=1)]
        
        if len(candidates) == 0:
            return self._sample_points(batch_size)
        
        acq_vals = self._acquisition_function(candidates)
        next_point = candidates[acq_vals.argmax()]
        return next_point.reshape(1, -1)

    def _evaluate_points(self, func, X):
        n_points = min(self.budget - self.n_evals, X.shape[0])
        y = np.array([func(x) for x in X[:n_points]])
        self.n_evals += n_points
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(self, func):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        while self.n_evals < self.budget:
            model = self._fit_model(self.X, self.y)
            X_next = self._select_next_points(1)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)
        
        best_f = self.y.min()
        best_x = self.X[self.y.argmin()]
        return best_f, best_x