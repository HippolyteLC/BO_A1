import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
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

        # Initial Sobol design
        sobol_engine = SobolEngine(dimension=self.dim, scramble=True, seed=self.random_seed)
        X_init = sobol_engine.draw(self.n_init).numpy()
        X_scaled = lb + X_init * (ub - lb)
        Y_init = np.array([float(func(x.tolist())) for x in X_scaled])

        # Normalize to unit hypercube
        X_normalized = normalize(torch.tensor(X_scaled, dtype=torch.float64), bounds=torch.tensor([lb, ub], dtype=torch.float64))

        # Initialize GP model
        covar_module = MaternKernel(nu=2.5, ard_num_dims=self.dim)
        model = SingleTaskGP(X_normalized, torch.tensor(-Y_init, dtype=torch.float64).unsqueeze(-1), covar_module=covar_module)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Fit GP model
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"GP fitting failed: {e}")
            return np.inf, np.zeros(self.dim)

        # Optimization loop
        n_evals = self.n_init
        best_f = np.min(Y_init)
        best_x = X_scaled[np.argmin(Y_init)]

        while n_evals < self.budget:
            # Update model
            try:
                fit_gpytorch_mll(mll)
            except Exception as e:
                print(f"GP fitting failed: {e}")
                # Fallback: use the best point perturbation
                perturbation = np.random.normal(0, 0.1, size=self.dim)
                x_perturbed_scaled = np.clip(best_x + perturbation, lb, ub)
                y_perturbed = float(func(x_perturbed_scaled.tolist()))
                n_evals += 1
                X_scaled = np.vstack([X_scaled, x_perturbed_scaled])
                Y_init = np.append(Y_init, y_perturbed)
                X_normalized = normalize(torch.tensor(X_scaled, dtype=torch.float64), bounds=torch.tensor([lb, ub], dtype=torch.float64))
                model.set_train_data(X_normalized, torch.tensor(-Y_init, dtype=torch.float64).unsqueeze(-1), strict=False)
                if y_perturbed < best_f:
                    best_f = y_perturbed
                    best_x = x_perturbed_scaled
                continue

            # Dynamic exploration weight
            exploration_weight = max(0.1, 0.5 - (n_evals - self.n_init) / (self.budget - self.n_init) * 0.4)

            # Acquisition function
            acq_func = qUpperConfidenceBound(model, beta=exploration_weight)
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
                # Fallback: use the best point perturbation
                perturbation = np.random.normal(0, 0.1, size=self.dim)
                x_perturbed_scaled = np.clip(best_x + perturbation, lb, ub)
                y_perturbed = float(func(x_perturbed_scaled.tolist()))
                n_evals += 1
                X_scaled = np.vstack([X_scaled, x_perturbed_scaled])
                Y_init = np.append(Y_init, y_perturbed)
                X_normalized = normalize(torch.tensor(X_scaled, dtype=torch.float64), bounds=torch.tensor([lb, ub], dtype=torch.float64))
                model.set_train_data(X_normalized, torch.tensor(-Y_init, dtype=torch.float64).unsqueeze(-1), strict=False)
                if y_perturbed < best_f:
                    best_f = y_perturbed
                    best_x = x_perturbed_scaled
                continue

            # Evaluate new candidate
            x_new_scaled = unnormalize(candidates, bounds=torch.tensor([lb, ub], dtype=torch.float64)).numpy()[0]
            y_new = float(func(x_new_scaled.tolist()))
            n_evals += 1

            # Update data
            X_scaled = np.vstack([X_scaled, x_new_scaled])
            Y_init = np.append(Y_init, y_new)
            X_normalized = normalize(torch.tensor(X_scaled, dtype=torch.float64), bounds=torch.tensor([lb, ub], dtype=torch.float64))

            # Update model
            model.set_train_data(X_normalized, torch.tensor(-Y_init, dtype=torch.float64).unsqueeze(-1), strict=False)

            # Update best found
            if y_new < best_f:
                best_f = y_new
                best_x = x_new_scaled

            # Adaptive trust-region refinement
            if n_evals < self.budget - 1 and y_new >= best_f:
                # Perturb the best point and evaluate
                perturbation = np.random.normal(0, 0.1, size=self.dim)
                x_perturbed_scaled = np.clip(best_x + perturbation, lb, ub)
                y_perturbed = float(func(x_perturbed_scaled.tolist()))
                n_evals += 1

                # Update data
                X_scaled = np.vstack([X_scaled, x_perturbed_scaled])
                Y_init = np.append(Y_init, y_perturbed)
                X_normalized = normalize(torch.tensor(X_scaled, dtype=torch.float64), bounds=torch.tensor([lb, ub], dtype=torch.float64))

                # Update model
                model.set_train_data(X_normalized, torch.tensor(-Y_init, dtype=torch.float64).unsqueeze(-1), strict=False)

                # Update best found
                if y_perturbed < best_f:
                    best_f = y_perturbed
                    best_x = x_perturbed_scaled

        return best_f, best_x