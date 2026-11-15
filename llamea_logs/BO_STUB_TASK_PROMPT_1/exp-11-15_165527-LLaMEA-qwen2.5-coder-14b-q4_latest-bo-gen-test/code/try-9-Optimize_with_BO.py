import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.models import SingleTaskGP
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

        # Initial Sobol design
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.random_seed)
        X_init = sobol.draw(self.n_init).numpy()
        X_scaled = unnormalize(torch.tensor(X_init, dtype=torch.float64), bounds)
        Y_init = np.array([float(func(x_scaled_np.tolist())) for x_scaled_np in X_scaled])
        Y_init = -Y_init  # Convert to maximization problem
        n_evals = self.n_init

        # Normalize initial data
        X_train_normalized = normalize(torch.tensor(X_scaled, dtype=torch.float64), bounds)
        Y_train_normalized = torch.tensor(Y_init, dtype=torch.float64).unsqueeze(-1)

        # Initialize model and MLL
        model = SingleTaskGP(X_train_normalized, Y_train_normalized, covar_module=MaternKernel(nu=2.5, ard_num_dims=self.dim))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Optimization loop
        best_f_max = float(Y_train_normalized.max().item())
        best_x = X_scaled[Y_init.argmin()]

        while n_evals < self.budget:
            try:
                # Fit the model
                fit_gpytorch_mll(mll)

                # Create acquisition function
                qkEI = qLogExpectedImprovement(model, best_f=best_f_max)

                # Optimize the acquisition function
                candidates, _ = optimize_acqf(
                    qkEI,
                    bounds=torch.tensor([[0.0] * self.dim, [1.0] * self.dim], dtype=torch.float64),
                    q=1,
                    num_restarts=5,
                    raw_samples=64,
                )

                # Evaluate the new candidate
                x_new_scaled = unnormalize(candidates, bounds).numpy()[0]
                y_new = float(func(x_new_scaled.tolist()))
                y_new = -y_new  # Convert to maximization problem
                n_evals += 1

                # Update training data
                X_train_normalized = torch.cat([X_train_normalized, candidates])
                Y_train_normalized = torch.cat([Y_train_normalized, torch.tensor([[y_new]], dtype=torch.float64)])

                # Update model and MLL
                model.set_train_data(X_train_normalized, Y_train_normalized, strict=False)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)

                # Update best found
                if y_new > best_f_max:
                    best_f_max = y_new
                    best_x = x_new_scaled

                # Adaptive trust-region refinement
                if y_new <= best_f_max:
                    # Perturb the best point and evaluate
                    perturbation = np.random.normal(0, 0.1, size=self.dim)
                    x_perturbed = best_x + perturbation
                    x_perturbed = np.clip(x_perturbed, lb, ub)
                    y_perturbed = float(func(x_perturbed.tolist()))
                    y_perturbed = -y_perturbed  # Convert to maximization problem
                    n_evals += 1

                    # Update training data
                    X_train_normalized = torch.cat([X_train_normalized, normalize(torch.tensor([x_perturbed], dtype=torch.float64), bounds)])
                    Y_train_normalized = torch.cat([Y_train_normalized, torch.tensor([[y_perturbed]], dtype=torch.float64)])

                    # Update model and MLL
                    model.set_train_data(X_train_normalized, Y_train_normalized, strict=False)
                    mll = ExactMarginalLogLikelihood(model.likelihood, model)

                    # Update best found
                    if y_perturbed > best_f_max:
                        best_f_max = y_perturbed
                        best_x = x_perturbed

            except Exception as e:
                # Fallback to a random candidate
                x_new_scaled = unnormalize(torch.rand((1, self.dim), dtype=torch.float64), bounds).numpy()[0]
                y_new = float(func(x_new_scaled.tolist()))
                y_new = -y_new  # Convert to maximization problem
                n_evals += 1

                # Update training data
                X_train_normalized = torch.cat([X_train_normalized, torch.tensor([x_new_scaled], dtype=torch.float64)])
                Y_train_normalized = torch.cat([Y_train_normalized, torch.tensor([[y_new]], dtype=torch.float64)])

                # Update model and MLL
                model.set_train_data(X_train_normalized, Y_train_normalized, strict=False)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)

                # Update best found
                if y_new > best_f_max:
                    best_f_max = y_new
                    best_x = x_new_scaled

        # Convert back to minimization scale
        best_f = -best_f_max
        return best_f, best_x