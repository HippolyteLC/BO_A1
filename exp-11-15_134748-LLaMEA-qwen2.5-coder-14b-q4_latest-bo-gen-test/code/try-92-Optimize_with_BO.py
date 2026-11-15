import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel

class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int, random_seed: int):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)
        self.best_f = np.inf
        self.best_x = None

    def __call__(self, func):
        lb = np.asarray(func.bounds.lb, dtype=float).reshape(-1)
        ub = np.asarray(func.bounds.ub, dtype=float).reshape(-1)

        # Generate initial design
        X = self.rng.uniform(lb, ub, size=(self.initial_points, self.dim))
        Y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.best_f = np.min(Y)
        self.best_x = X[np.argmin(Y)]

        # Convert to PyTorch tensors
        X_train = torch.tensor(X, dtype=torch.float)
        Y_train = torch.tensor(Y, dtype=torch.float)

        # Initialize model and likelihood
        kernel = ScaleKernel(RBFKernel())
        model = SingleTaskGP(X_train, Y_train, kernel=kernel)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Fit the model
        fit_gpytorch_model(mll)

        # Optimization loop
        for _ in range(self.budget - self.initial_points):
            # Define acquisition function
            ei = ExpectedImprovement(model, best_f=self.best_f)

            # Optimize the acquisition function
            candidates, _ = optimize_acqf(
                ei,
                bounds=torch.tensor([lb, ub], dtype=torch.float),
                q=1,
                num_restarts=5,
                raw_samples=20,
                options={"batch_limit": 5},
            )

            # Evaluate the new candidate
            new_x = candidates.detach().numpy().flatten()
            new_f = func(new_x)

            # Update the best found
            if new_f < self.best_f:
                self.best_f = new_f
                self.best_x = new_x

            # Update training data
            X_train = torch.cat([X_train, candidates])
            Y_train = torch.cat([Y_train, torch.tensor([[new_f]], dtype=torch.float)])

            # Refit the model
            model.set_train_data(X_train, Y_train, strict=False)
            fit_gpytorch_model(mll)

        return self.best_f, self.best_x