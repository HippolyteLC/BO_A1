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
        X_scaled = lb + np.clip(X_unit, 0.0, 1.0) * (ub - lb)
        Y = np.array([float(func(x.tolist())) for x in X_scaled], dtype=np.float64).reshape(-1, 1)

        # Normalize to unit hypercube
        X_normalized = normalize(torch.tensor(X_scaled, dtype=torch.float64), bounds=bounds)

        # Initialize GP model
        covar_module = MaternKernel(nu=2.5, ard_num_dims=self.dim)
        model = SingleTaskGP(X_normalized, torch.tensor(Y, dtype=torch.float64), covar_module=covar_module)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Fit GP model
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"GP fitting failed: {e}")
            return float('inf'), np.zeros(self.dim)

        # Track best values
        best_f = float(Y.min())
        best_x = X_scaled[Y.argmin()]

        # Optimization loop
        n_evals = self.n_init
        beta = 0.1  # Initial exploration weight
        while n_evals < self.budget:
            # Acquisition function
            acq_func = qLogExpectedImprovement(model, best_f=best_f)
            try:
                candidates, _ = optimize_acqf(
                    acq_func,
                    bounds=torch.tensor([[0.0] * self.dim, [1.0] * self.dim], dtype=torch.float64),
                    q=1,
                    num_restarts=5,
                    raw_samples=64,
                    options={"maxiter": 200, "init_points": 20, "beta": beta}
                )
            except Exception as e:
                print(f"Acquisition optimization failed: {e}")
                # Fallback to a random point
                x_unit = sobol_engine.draw(1).numpy()
                candidates = torch.tensor(lb + np.clip(x_unit, 0.0, 1.0) * (ub - lb), dtype=torch.float64)

            # Evaluate new candidate
            x_candidate_scaled = candidates.numpy().flatten()
            y_candidate = float(func(x_candidate_scaled.tolist()))
            Y = np.append(Y, y_candidate).reshape(-1, 1)

            # Update best values
            if y_candidate < best_f:
                best_f = y_candidate
                best_x = x_candidate_scaled

            # Update GP model
            X_normalized = torch.cat([X_normalized, normalize(candidates, bounds=bounds)], dim=0)
            model.set_train_data(X_normalized, torch.tensor(Y, dtype=torch.float64), strict=False)
            try:
                fit_gpytorch_mll(mll)
            except Exception as e:
                print(f"GP refitting failed: {e}")
                return float('inf'), np.zeros(self.dim)

            # Trust-region refinement if improvement stalls
            if n_evals > self.n_init and y_candidate >= best_f:
                # Refine around the best point
                lb_refine = np.clip(best_x - 0.1, lb, ub)
                ub_refine = np.clip(best_x + 0.1, lb, ub)
                bounds_refine = torch.tensor([lb_refine, ub_refine], dtype=torch.float64)
                sobol_engine_refine = SobolEngine(dimension=self.dim, scramble=True, seed=self.random_seed)
                X_unit_refine = sobol_engine_refine.draw(10).numpy()
                X_scaled_refine = lb_refine + np.clip(X_unit_refine, 0.0, 1.0) * (ub_refine - lb_refine)
                Y_refine = np.array([float(func(x.tolist())) for x in X_scaled_refine], dtype=np.float64).reshape(-1, 1)
                X_normalized_refine = normalize(torch.tensor(X_scaled_refine, dtype=torch.float64), bounds=bounds_refine)
                model.set_train_data(X_normalized_refine, torch.tensor(Y_refine, dtype=torch.float64), strict=False)
                try:
                    fit_gpytorch_mll(mll)
                except Exception as e:
                    print(f"GP refitting failed: {e}")
                    return float('inf'), np.zeros(self.dim)

                # Update best values from refinement
                if Y_refine.min() < best_f:
                    best_f = Y_refine.min()
                    best_x = X_scaled_refine[Y_refine.argmin()]

            # Adjust exploration weight dynamically
            if n_evals % 10 == 0:
                beta *= 0.9

            n_evals += 1

        return best_f, best_x