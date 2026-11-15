import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel
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
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.random_seed)
        X_init = sobol.draw(self.n_init).numpy()
        X_init = unnormalize(torch.tensor(X_init, dtype=torch.float64), bounds)

        # Evaluate initial points
        Y_init = []
        for x in X_init:
            y_val = float(func(x.tolist()))
            Y_init.append(y_val)
        Y_init = torch.tensor(Y_init, dtype=torch.float64).reshape(-1, 1)

        # Convert to maximization target
        train_Y = -Y_init

        # Initialize GP model
        covar_module = MaternKernel(nu=2.5, ard_num_dims=self.dim)
        model = SingleTaskGP(X_init, train_Y, covar_module=covar_module)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Fit GP model
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"GP fitting failed: {e}")
            return float('inf'), None

        # Track best values
        best_f_max = float(train_Y.max().item())
        best_x = X_init[train_Y.argmax()].numpy()

        # Optimization loop
        n_evals = self.n_init
        while n_evals < self.budget:
            # Acquisition function
            acq_func = qLogExpectedImprovement(model, best_f_max)
            try:
                candidates, _ = optimize_acqf(
                    acq_func,
                    bounds=bounds,
                    q=1,
                    num_restarts=5,
                    raw_samples=64,
                    options={"maxiter": 200}
                )
            except Exception as e:
                print(f"Acquisition optimization failed: {e}")
                return float('inf'), None

            # Evaluate new candidate
            x_new = candidates.numpy()
            y_new = float(func(x_new.tolist()))
            Y_new = torch.tensor([-y_new], dtype=torch.float64).reshape(-1, 1)

            # Update model
            X_init = torch.cat([X_init, candidates])
            train_Y = torch.cat([train_Y, Y_new])

            # Refit GP model
            try:
                model.set_train_data(X_init, train_Y, strict=False)
                fit_gpytorch_mll(mll)
            except Exception as e:
                print(f"GP refitting failed: {e}")
                return float('inf'), None

            # Update best values
            best_f_max = float(train_Y.max().item())
            if -y_new < -train_Y.min().item():
                best_x = x_new

            n_evals += 1

        return -best_f_max.item(), best_x