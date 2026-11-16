import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
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

        # Initial design using Sobol quasi-random sampling
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.random_seed)
        X_init = sobol.draw(self.n_init).numpy()
        X_scaled_init = lb + X_init * (ub - lb)
        Y_init = np.array([float(func(x.tolist())) for x in X_scaled_init])

        # Normalize initial data to unit hypercube
        X_normalized_init = normalize(torch.tensor(X_scaled_init, dtype=torch.float64), bounds=bounds)

        # Initialize GP model
        covar_module = MaternKernel(nu=2.5, ard_num_dims=self.dim)
        model = SingleTaskGP(X_normalized_init, torch.tensor(Y_init, dtype=torch.float64).unsqueeze(-1), covar_module=covar_module)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Fit the GP model
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"Error fitting GP: {e}")
            return float('inf'), np.zeros(self.dim)

        # Track best observed value
        best_f = float('inf')
        best_x = np.zeros(self.dim)

        # Optimization loop
        n_evals = self.n_init
        while n_evals < self.budget:
            # Acquisition function
            qLEI = qLogExpectedImprovement(model, torch.tensor([-best_f], dtype=torch.float64).unsqueeze(-1), beta=0.1)

            # Optimize acquisition function
            try:
                candidates, _ = optimize_acqf(
                    qLEI,
                    bounds=torch.tensor([[0.0] * self.dim, [1.0] * self.dim], dtype=torch.float64),
                    q=1,
                    num_restarts=5,
                    raw_samples=64,
                )
            except Exception as e:
                print(f"Error optimizing acquisition function: {e}")
                return float('inf'), np.zeros(self.dim)

            # Unnormalize and evaluate new candidate
            x_candidate = unnormalize(candidates, bounds).numpy().flatten()
            y_candidate = float(func(x_candidate.tolist()))

            # Update best observed value
            if y_candidate < best_f:
                best_f = y_candidate
                best_x = x_candidate

            # Update training data
            X_normalized = torch.cat([X_normalized_init, normalize(torch.tensor([x_candidate], dtype=torch.float64), bounds=bounds)])
            Y = np.append(Y_init, y_candidate)

            # Refit the GP model
            try:
                model.set_train_data(X_normalized, torch.tensor(Y, dtype=torch.float64).unsqueeze(-1), strict=False)
                fit_gpytorch_mll(mll)
            except Exception as e:
                print(f"Error refitting GP: {e}")
                return float('inf'), np.zeros(self.dim)

            n_evals += 1

        return best_f, best_x