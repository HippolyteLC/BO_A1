import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from botorch.utils.sampling import draw_sobol_samples
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
        X_init = draw_sobol_samples(sobol_engine, self.n_init, dtype=torch.float64).squeeze(0)
        X_init_scaled = unnormalize(X_init, bounds)

        Y_init = []
        for x_scaled_np in X_init_scaled:
            y_val = float(func(x_scaled_np.tolist()))
            Y_init.append(y_val)
        Y_init = torch.tensor(Y_init, dtype=torch.float64).reshape(-1, 1)

        train_X = X_init
        train_Y = -Y_init  # Convert to maximization

        model = SingleTaskGP(train_X, train_Y, covar_module=MaternKernel(nu=2.5, ard_num_dims=self.dim))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"Error fitting GP: {e}")
            return np.inf, None

        best_f_max = float(train_Y.max().item())
        best_x = X_init_scaled[torch.argmin(Y_init)]

        n_evals = self.n_init
        while n_evals < self.budget:
            acq_func = qLogExpectedImprovement(model, best_f_max, beta=0.1)
            try:
                candidates, _ = optimize_acqf(
                    acq_func,
                    bounds=torch.tensor([[0.0] * self.dim, [1.0] * self.dim], dtype=torch.float64),
                    q=1,
                    num_restarts=5,
                    raw_samples=64,
                )
            except Exception as e:
                print(f"Error optimizing acquisition function: {e}")
                return np.inf, None

            candidate_scaled = unnormalize(candidates, bounds)
            y_val = float(func(candidate_scaled.squeeze().tolist()))
            Y_init = torch.cat([Y_init, torch.tensor([[y_val]], dtype=torch.float64)])
            train_Y = -Y_init

            train_X = torch.cat([train_X, candidates])

            model.set_train_data(train_X, train_Y, strict=False)
            try:
                fit_gpytorch_mll(mll)
            except Exception as e:
                print(f"Error fitting GP: {e}")
                return np.inf, None

            best_f_max = float(train_Y.max().item())
            if y_val < float(Y_init.min().item()):
                best_x = candidate_scaled.squeeze()

            n_evals += 1

        return float(Y_init.min().item()), best_x.numpy()