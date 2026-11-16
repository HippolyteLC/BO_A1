import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine
from gpytorch.kernels import MaternKernel

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

        # Initial design
        sobol_engine = SobolEngine(dimension=self.dim, scramble=True, seed=self.random_seed)
        X_init = sobol_engine.draw(self.n_init).detach().numpy()
        X_scaled_init = lb + X_init * (ub - lb)
        Y_init = np.array([float(func(x_scaled.tolist())) for x_scaled in X_scaled_init])

        # Normalize initial design
        X_normalized_init = normalize(torch.tensor(X_scaled_init, dtype=torch.float64), bounds=bounds)

        # Initialize GP model
        model = SingleTaskGP(X_normalized_init, torch.tensor(Y_init, dtype=torch.float64).unsqueeze(-1), covar_module=MaternKernel(nu=2.5, ard_num_dims=self.dim))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Fit GP model
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            # Fallback to a perturbed best point or Sobol candidate
            X_normalized_init = SobolEngine(dimension=self.dim, scramble=True, seed=self.random_seed).draw(1).detach().numpy()
            Y_init = np.array([float(func(x_scaled.tolist())) for x_scaled in X_scaled_init])
            X_normalized_init = normalize(torch.tensor(X_scaled_init, dtype=torch.float64), bounds=bounds)
            model = SingleTaskGP(X_normalized_init, torch.tensor(Y_init, dtype=torch.float64).unsqueeze(-1), covar_module=MaternKernel(nu=2.5, ard_num_dims=self.dim))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

        # Optimization loop
        n_evals = self.n_init
        best_f = float('inf')
        best_x = None

        while n_evals < self.budget:
            # Update model
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            try:
                fit_gpytorch_mll(mll)
            except Exception:
                # Fallback to a perturbed best point or Sobol candidate
                X_normalized_new = SobolEngine(dimension=self.dim, scramble=True, seed=self.random_seed).draw(1).detach().numpy()
                Y_new = np.array([float(func(x_scaled.tolist())) for x_scaled in X_scaled_init])
                X_normalized_new = normalize(torch.tensor(X_scaled_init, dtype=torch.float64), bounds=bounds)
                model = SingleTaskGP(X_normalized_new, torch.tensor(Y_new, dtype=torch.float64).unsqueeze(-1), covar_module=MaternKernel(nu=2.5, ard_num_dims=self.dim))
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)

            # Acquisition function
            best_f_max = float(-model.train_targets.min().item())
            qEI = ExpectedImprovement(model, torch.tensor([best_f_max], dtype=torch.float64).unsqueeze(-1))
            candidates, _ = optimize_acqf(
                qEI,
                bounds=bounds,
                q=1,
                num_restarts=5,
                raw_samples=64,
            )

            # Evaluate new candidate
            X_new = candidates.detach().numpy()
            Y_new = np.array([float(func(x_scaled.tolist())) for x_scaled in X_new])

            # Update best found
            if Y_new[0] < best_f:
                best_f = Y_new[0]
                best_x = X_new[0]

            # Update model with new data
            X_normalized_new = normalize(torch.tensor(X_new, dtype=torch.float64), bounds=bounds)
            Y_normalized_new = torch.tensor(Y_new, dtype=torch.float64).unsqueeze(-1)
            model.train_inputs = (torch.cat([model.train_inputs[0], X_normalized_new]),)
            model.train_targets = torch.cat([model.train_targets, Y_normalized_new])

            n_evals += 1

        return best_f, best_x