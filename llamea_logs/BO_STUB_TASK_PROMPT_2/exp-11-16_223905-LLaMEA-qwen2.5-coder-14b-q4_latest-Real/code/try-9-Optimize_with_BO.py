import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.kernels import MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
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
        X_init_scaled = lb + X_init * (ub - lb)
        Y_init = np.array([float(func(x.tolist())) for x in X_init_scaled])

        # Normalize initial data
        X_train_normalized = normalize(torch.tensor(X_init, dtype=torch.float64), bounds=bounds)
        Y_train = torch.tensor(Y_init, dtype=torch.float64).unsqueeze(-1)

        # Initialize model
        likelihood = GaussianLikelihood(noise_constraint=None)
        covar_module = MaternKernel(nu=2.5, ard_num_dims=self.dim)
        model = SingleTaskGP(X_train_normalized, Y_train, covar_module=covar_module, likelihood=likelihood)
        mll = ExactMarginalLogLikelihood(likelihood, model)

        # Fit the model
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"Model fitting failed: {e}")
            return float('inf'), np.zeros(self.dim)

        best_f_max = float(-Y_train.min().item())
        best_x = X_init_scaled[Y_init.argmin()]

        n_evals = self.n_init

        while n_evals < self.budget:
            # Acquisition function
            qEI = qLogExpectedImprovement(model, best_f=best_f_max)
            try:
                candidates, _ = optimize_acqf(
                    qEI,
                    bounds=torch.tensor([[0.0] * self.dim, [1.0] * self.dim], dtype=torch.float64),
                    q=1,
                    num_restarts=5,
                    raw_samples=64,
                )
            except Exception as e:
                print(f"Acquisition optimization failed: {e}")
                candidates = torch.tensor(X_init[np.random.randint(self.n_init)], dtype=torch.float64).unsqueeze(0)

            # Evaluate the new candidate
            candidate_scaled = unnormalize(candidates, bounds=bounds).numpy()[0]
            y_val = float(func(candidate_scaled.tolist()))
            n_evals += 1

            # Update training data
            X_train_normalized = torch.cat([X_train_normalized, candidates], dim=0)
            Y_train = torch.cat([Y_train, torch.tensor([[y_val]], dtype=torch.float64)], dim=0)

            # Update model
            model = SingleTaskGP(X_train_normalized, Y_train, covar_module=covar_module, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(likelihood, model)
            try:
                fit_gpytorch_mll(mll)
            except Exception as e:
                print(f"Model refitting failed: {e}")
                return float('inf'), np.zeros(self.dim)

            best_f_max = float(-Y_train.min().item())
            if y_val < Y_init.min():
                best_x = candidate_scaled
                Y_init = np.append(Y_init, y_val)

            # Adaptive trust-region refinement
            if n_evals < self.budget - 10 and n_evals % 5 == 0:
                try:
                    # Local refinement around the best point
                    local_bounds = torch.tensor([
                        np.clip(best_x - 0.1, lb, ub),
                        np.clip(best_x + 0.1, lb, ub)
                    ], dtype=torch.float64)
                    local_candidates, _ = optimize_acqf(
                        qEI,
                        bounds=local_bounds,
                        q=1,
                        num_restarts=5,
                        raw_samples=64,
                    )
                    local_candidate_scaled = unnormalize(local_candidates, bounds=bounds).numpy()[0]
                    local_y_val = float(func(local_candidate_scaled.tolist()))
                    n_evals += 1

                    if local_y_val < y_val:
                        X_train_normalized = torch.cat([X_train_normalized, local_candidates], dim=0)
                        Y_train = torch.cat([Y_train, torch.tensor([[local_y_val]], dtype=torch.float64)], dim=0)
                        model = SingleTaskGP(X_train_normalized, Y_train, covar_module=covar_module, likelihood=likelihood)
                        mll = ExactMarginalLogLikelihood(likelihood, model)
                        fit_gpytorch_mll(mll)
                        best_f_max = float(-Y_train.min().item())
                        if local_y_val < Y_init.min():
                            best_x = local_candidate_scaled
                            Y_init = np.append(Y_init, local_y_val)
                except Exception as e:
                    print(f"Local refinement failed: {e}")

        return float(best_f_max), best_x