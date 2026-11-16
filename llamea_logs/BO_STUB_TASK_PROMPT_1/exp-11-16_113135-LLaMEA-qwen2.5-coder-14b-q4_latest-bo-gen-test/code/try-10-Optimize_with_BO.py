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

            # Trust-region refinement
            if n_evals < self.budget - 1:
                try:
                    # Define a small trust region around the best point
                    trust_region = 0.1
                    lower_bound = np.clip(best_x - trust_region, lb, ub)
                    upper_bound = np.clip(best_x + trust_region, lb, ub)
                    trust_region_bounds = torch.tensor([lower_bound, upper_bound], dtype=torch.float64)

                    # Optimize within the trust region
                    trust_region_acq_func = qLogExpectedImprovement(model, best_f=best_f)
                    trust_region_candidates, _ = optimize_acqf(
                        trust_region_acq_func,
                        bounds=trust_region_bounds,
                        q=1,
                        num_restarts=5,
                        raw_samples=64,
                    )

                    # Evaluate new candidate in the trust region
                    x_trust_region_scaled = trust_region_candidates.numpy().flatten()
                    y_trust_region = float(func(x_trust_region_scaled.tolist()))
                    Y = np.append(Y, y_trust_region).reshape(-1, 1)

                    # Update best values if improved
                    if y_trust_region < best_f:
                        best_f = y_trust_region
                        best_x = x_trust_region_scaled

                    # Update GP model with trust region candidate
                    X_normalized = torch.cat([X_normalized, normalize(trust_region_candidates, bounds=bounds)], dim=0)
                    model.set_train_data(X_normalized, torch.tensor(Y, dtype=torch.float64), strict=False)
                    fit_gpytorch_mll(mll)
                except Exception as e:
                    print(f"Trust-region optimization failed: {e}")

            n_evals += 1

        return best_f, best_x