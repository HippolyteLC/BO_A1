import numpy as np
import torch
from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
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
        X_scaled = lb + X_init * (ub - lb)
        Y_init = np.array([float(func(x.tolist())) for x in X_scaled])

        train_X = torch.tensor(X_init, dtype=torch.float64)
        train_Y = torch.tensor(Y_init, dtype=torch.float64).unsqueeze(-1)

        # Composite kernel
        kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim) + RBFKernel(ard_num_dims=self.dim))
        model = SingleTaskGP(train_X, train_Y, covar_module=kernel)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"GP fitting failed: {e}")
            return float('inf'), np.zeros(self.dim)

        best_f_max = float(train_Y.max().item())
        best_x = X_scaled[np.argmin(Y_init)]

        n_evals = self.n_init
        while n_evals < self.budget:
            qEI = qLogExpectedImprovement(model, best_f=best_f_max)
            try:
                candidates, _ = optimize_acqf(
                    qEI,
                    bounds=bounds,
                    q=1,
                    num_restarts=5,
                    raw_samples=64,
                )
            except Exception as e:
                print(f"Acquisition optimization failed: {e}")
                candidates = torch.tensor([best_x], dtype=torch.float64)

            candidate_scaled = unnormalize(candidates, bounds).numpy()
            candidate_f = float(func(candidate_scaled[0].tolist()))

            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, torch.tensor([[candidate_f]], dtype=torch.float64)])

            model.set_train_data(train_X, train_Y, strict=False)
            try:
                fit_gpytorch_mll(mll)
            except Exception as e:
                print(f"GP fitting failed: {e}")
                return float('inf'), np.zeros(self.dim)

            best_f_max = float(train_Y.max().item())
            if candidate_f < best_f_max:
                best_f_max = candidate_f
                best_x = candidate_scaled[0]

            n_evals += 1

        return float(best_f_max), best_x