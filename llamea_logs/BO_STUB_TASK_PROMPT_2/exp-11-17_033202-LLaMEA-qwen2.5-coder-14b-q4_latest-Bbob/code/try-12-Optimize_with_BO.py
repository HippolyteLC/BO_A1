import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.kernels import MaternKernel
from gotorch.likelihoods import GaussianLikelihood
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
        sobol_engine = SobolEngine(dimension=self.dim, scramble=True, seed=self.random_seed)
        X_init = sobol_engine.draw(self.n_init).numpy()
        X_train = torch.tensor(X_init, dtype=torch.float64)
        Y_train = []

        for x in X_train.numpy():
            y_val = float(func(x.tolist()))
            Y_train.append(y_val)

        Y_train = torch.tensor(Y_train, dtype=torch.float64).unsqueeze(-1)
        model = SingleTaskGP(X_train, Y_train, covar_module=MaternKernel(nu=2.5, ard_num_dims=self.dim))
        likelihood = GaussianLikelihood()
        mll = ExactMarginalLogLikelihood(likelihood, model)

        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"Error fitting GP: {e}")
            return float('inf'), np.zeros(self.dim)

        best_f = float(Y_train.min().item())
        best_x = X_train[Y_train.argmin()].numpy()

        n_evals = self.n_init
        while n_evals < self.budget:
            qEI = qLogExpectedImprovement(model, best_f=best_f)
            candidates, _ = optimize_acqf(
                qEI,
                bounds=bounds,
                q=1,
                num_restarts=5,
                raw_samples=64,
            )

            candidate = candidates[0].numpy()
            y_val = float(func(candidate.tolist()))
            Y_train = torch.cat([Y_train, torch.tensor([[y_val]], dtype=torch.float64)])
            X_train = torch.cat([X_train, torch.tensor([candidate], dtype=torch.float64)])

            try:
                model = SingleTaskGP(X_train, Y_train, covar_module=MaternKernel(nu=2.5, ard_num_dims=self.dim))
                likelihood = GaussianLikelihood()
                mll = ExactMarginalLogLikelihood(likelihood, model)
                fit_gpytorch_mll(mll)
            except Exception as e:
                print(f"Error updating GP: {e}")
                # Fallback to the best known point
                candidate = best_x
                y_val = float(func(candidate.tolist()))
                Y_train = torch.cat([Y_train, torch.tensor([[y_val]], dtype=torch.float64)])
                X_train = torch.cat([X_train, torch.tensor([candidate], dtype=torch.float64)])

            if y_val < best_f:
                best_f = y_val
                best_x = candidate

            n_evals += 1

        return best_f, best_x