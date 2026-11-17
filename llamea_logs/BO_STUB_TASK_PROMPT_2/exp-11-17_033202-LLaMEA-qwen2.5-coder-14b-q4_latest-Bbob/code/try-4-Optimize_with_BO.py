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

        # Initialize Sobol engine
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.random_seed)
        X_init = sobol.draw(self.n_init).numpy()

        # Normalize initial points
        X_init_normalized = normalize(torch.tensor(X_init, dtype=torch.float64), bounds=torch.tensor([lb, ub], dtype=torch.float64))

        # Evaluate initial points
        Y_init = []
        for x in X_init_normalized.numpy():
            y_val = float(func(x.tolist()))
            Y_init.append(y_val)
            if len(Y_init) >= self.budget:
                break

        Y_init = np.array(Y_init)
        Y_init_max = Y_init.max()

        # Convert to maximization targets
        train_Y = -torch.tensor(Y_init, dtype=torch.float64).reshape(-1, 1)

        # Initialize GP model
        model = SingleTaskGP(X_init_normalized, train_Y, covar_module=MaternKernel(nu=2.5, ard_num_dims=self.dim))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Fit GP model
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"GP fitting failed: {e}")
            return float('inf'), np.zeros(self.dim)

        # Optimization loop
        n_evals = self.n_init
        best_f = -train_Y.min().item()
        best_x = X_init_normalized[train_Y.argmin()].numpy()

        while n_evals < self.budget:
            # Acquisition function
            acq_func = qLogExpectedImprovement(model, best_f=-train_Y.min().item())
            candidates, _ = optimize_acqf(
                acq_func,
                bounds=torch.tensor([[0.0] * self.dim, [1.0] * self.dim], dtype=torch.float64),
                q=1,
                num_restarts=5,
                raw_samples=64,
            )

            # Unnormalize candidate
            candidate = unnormalize(candidates, bounds=torch.tensor([lb, ub], dtype=torch.float64)).numpy()[0]

            # Evaluate candidate
            y_val = float(func(candidate.tolist()))
            n_evals += 1

            # Update best found
            if y_val < best_f:
                best_f = y_val
                best_x = candidate

            # Update GP model
            train_X = torch.cat([X_init_normalized, candidates], dim=0)
            train_Y = torch.cat([train_Y, -torch.tensor([[y_val]], dtype=torch.float64)], dim=0)
            model = SingleTaskGP(train_X, train_Y, covar_module=MaternKernel(nu=2.5, ard_num_dims=self.dim))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            try:
                fit_gpytorch_mll(mll)
            except Exception as e:
                print(f"GP fitting failed: {e}")
                return float('inf'), np.zeros(self.dim)

        return best_f, best_x