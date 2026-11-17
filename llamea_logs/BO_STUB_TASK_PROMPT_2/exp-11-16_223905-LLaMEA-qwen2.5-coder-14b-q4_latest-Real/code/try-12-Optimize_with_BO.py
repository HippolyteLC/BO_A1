import numpy as np
import torch
from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.kernels import MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int, random_seed: int):
        self.budget = budget
        self.dim = dim
        self.random_seed = random_seed
        self.n_init = min(initial_points, budget)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=np.float64).reshape(-1)
            ub = np.asarray(func.bounds.ub, dtype=np.float64).reshape(-1)
            if lb.size == 1:
                lb = np.full(self.dim, lb.item(), dtype=np.float64)
            if ub.size == 1:
                ub = np.full(self.dim, ub.item(), dtype=np.float64)
        else:
            lb = np.full(self.dim, -5.0, dtype=np.float64)
            ub = np.full(self.dim, 5.0, dtype=np.float64)

        bounds = torch.tensor([lb, ub], dtype=torch.float64)

        # Initial design using Sobol quasi-random sampling
        sobol_engine = SobolEngine(dimension=self.dim, scramble=True, seed=self.random_seed)
        X_init = sobol_engine.draw(n=self.n_init).numpy()
        X_init_scaled = lb + X_init * (ub - lb)
        Y_init = np.array([float(func(x.tolist())) for x in X_init_scaled])

        # Normalize initial data
        X_train_normalized = normalize(torch.tensor(X_init, dtype=torch.float64), bounds=bounds)
        Y_train = torch.tensor(Y_init, dtype=torch.float64).unsqueeze(-1)

        # Initialize GP model
        model = SingleTaskGP(X_train_normalized, Y_train, covar_module=MaternKernel(nu=2.5, ard_num_dims=self.dim))
        likelihood = GaussianLikelihood()
        mll = ExactMarginalLogLikelihood(likelihood, model)

        # Fit GP model
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"GP fitting failed: {e}")
            return float('inf'), np.zeros(self.dim)

        # Optimization loop
        n_evals = self.n_init
        best_f = float(Y_train.min().item())
        best_x = X_train_normalized[Y_train.argmin()].numpy()

        while n_evals < self.budget:
            # Acquisition function
            acqf = qLogExpectedImprovement(model, best_f=best_f)
            try:
                candidates, _ = optimize_acqf(
                    acqf,
                    bounds=torch.tensor([[0.0] * self.dim, [1.0] * self.dim], dtype=torch.float64),
                    q=1,
                    num_restarts=5,
                    raw_samples=64,
                )
            except Exception as e:
                print(f"Acquisition optimization failed: {e}")
                candidates = torch.rand((1, self.dim), dtype=torch.float64)

            # Unnormalize and evaluate new candidate
            x_candidate_scaled = unnormalize(candidates, bounds=bounds).numpy()[0]
            y_candidate = float(func(x_candidate_scaled.tolist()))

            # Update training data
            X_train_normalized = torch.cat([X_train_normalized, candidates], dim=0)
            Y_train = torch.cat([Y_train, torch.tensor([[y_candidate]], dtype=torch.float64)], dim=0)

            # Refit GP model
            try:
                model.set_train_data(X_train_normalized, Y_train, strict=False)
                fit_gpytorch_mll(mll)
            except Exception as e:
                print(f"GP refitting failed: {e}")
                return float('inf'), np.zeros(self.dim)

            # Update best found
            if y_candidate < best_f:
                best_f = y_candidate
                best_x = x_candidate_scaled

            n_evals += 1

        return best_f, best_x