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

        # Initial design using Sobol quasi-random sampling
        sobol_engine = SobolEngine(dimension=self.dim, scramble=True, seed=self.random_seed)
        X_init = sobol_engine.draw(self.n_init).numpy()
        X_init_scaled = unnormalize(torch.tensor(X_init, dtype=torch.float64), bounds)
        Y_init = np.array([float(func(x_scaled_np.tolist())) for x_scaled_np in X_init_scaled])

        # Normalize initial data
        X_normalized = normalize(torch.tensor(X_init, dtype=torch.float64), bounds)
        Y_normalized = -torch.tensor(Y_init, dtype=torch.float64).view(-1, 1)

        # Initialize model
        model = SingleTaskGP(X_normalized, Y_normalized, covar_module=MaternKernel(nu=2.5, ard_num_dims=self.dim))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Fit the model
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"Model fitting failed: {e}")
            return float('inf'), np.zeros(self.dim)

        # Optimization loop
        n_evals = self.n_init
        best_f = float('inf')
        best_x = None

        while n_evals < self.budget:
            # Update best_f
            best_f_max = float(-Y_normalized.min().item())
            if best_f > best_f_max:
                best_f = best_f_max
                best_x = X_init_scaled[Y_init.argmin()]

            # Acquisition function
            acq_func = qLogExpectedImprovement(model, best_f=best_f_max)

            # Optimize acquisition function
            try:
                candidates, _ = optimize_acqf(
                    acq_func,
                    bounds=torch.tensor([[0.0] * self.dim, [1.0] * self.dim], dtype=torch.float64),
                    q=1,
                    num_restarts=5,
                    raw_samples=64,
                )
            except Exception as e:
                print(f"Acquisition optimization failed: {e}")
                return float('inf'), np.zeros(self.dim)

            # Evaluate new candidate
            x_new_scaled = unnormalize(candidates, bounds)
            y_new = float(func(x_new_scaled.numpy().tolist()))
            Y_normalized = torch.cat([Y_normalized, -torch.tensor([[y_new]], dtype=torch.float64)])

            # Update model
            X_normalized = torch.cat([X_normalized, candidates])
            try:
                model.set_train_data(X_normalized, Y_normalized, strict=False)
                fit_gpytorch_mll(mll)
            except Exception as e:
                print(f"Model update failed: {e}")
                return float('inf'), np.zeros(self.dim)

            n_evals += 1

        return best_f, best_x