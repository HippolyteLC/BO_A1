import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.kernels import MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

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

        # Initial Sobol points
        sobol_engine = SobolEngine(dimension=self.dim, scramble=True, seed=self.random_seed)
        X_unit = sobol_engine.draw(self.n_init).numpy()
        X_scaled = lb + X_unit * (ub - lb)
        y_vals = [float(func(x.tolist())) for x in X_scaled]
        X_scaled = torch.tensor(X_scaled, dtype=torch.float64)
        y_vals = torch.tensor(y_vals, dtype=torch.float64).unsqueeze(-1)

        # Normalize to [0, 1]
        X_unit = normalize(X_scaled, bounds=bounds)

        # Initialize GP model
        model = SingleTaskGP(X_unit, y_vals, covar_module=MaternKernel(nu=2.5, ard_num_dims=self.dim))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Track best values
        best_f = float(y_vals.min().item())
        best_x = X_scaled[y_vals.argmin()]

        # Optimization loop
        n_evals = self.n_init
        while n_evals < self.budget:
            try:
                # Fit the GP model
                fit_gpytorch_mll(mll)

                # Acquisition function
                acq_func = qLogExpectedImprovement(model, best_f=best_f)

                # Optimize the acquisition function
                candidates, _ = optimize_acqf(
                    acq_func,
                    bounds=torch.tensor([[0.0] * self.dim, [1.0] * self.dim], dtype=torch.float64),
                    q=1,
                    num_restarts=5,
                    raw_samples=64,
                )

                # Unnormalize the candidate
                candidate_scaled = unnormalize(candidates, bounds=bounds).squeeze()
                candidate_val = float(func(candidate_scaled.tolist()))

                # Update the model with the new data
                X_unit = torch.cat([X_unit, candidates])
                y_vals = torch.cat([y_vals, torch.tensor([[candidate_val]], dtype=torch.float64)])
                model.set_train_data(X_unit, y_vals, strict=False)

                # Update best values
                if candidate_val < best_f:
                    best_f = candidate_val
                    best_x = candidate_scaled

                n_evals += 1

            except Exception as e:
                # Fallback to a random point
                candidate_scaled = torch.tensor(lb + np.random.rand(self.dim) * (ub - lb), dtype=torch.float64)
                candidate_val = float(func(candidate_scaled.tolist()))

                # Update the model with the new data
                X_unit = torch.cat([X_unit, normalize(candidate_scaled, bounds=bounds)])
                y_vals = torch.cat([y_vals, torch.tensor([[candidate_val]], dtype=torch.float64)])
                model.set_train_data(X_unit, y_vals, strict=False)

                # Update best values
                if candidate_val < best_f:
                    best_f = candidate_val
                    best_x = candidate_scaled

                n_evals += 1

        return best_f, best_x.numpy()