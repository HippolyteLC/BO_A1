import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.optim import conjugate_gradient

class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int, random_seed: int):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])

    def __call__(self, func):
        lb = torch.tensor(func.bounds.lb, dtype=torch.float).reshape(-1)
        ub = torch.tensor(func.bounds.ub, dtype=torch.float).reshape(-1)

        # Initial random sampling
        X = self.rng.uniform(lb.numpy(), ub.numpy(), size=(self.initial_points, self.dim))
        Y = np.array([func(x) for x in X])

        # Normalize inputs
        X_normalized = (X - lb.numpy()) / (ub.numpy() - lb.numpy())

        # Convert to tensors
        X_tensor = torch.tensor(X_normalized, dtype=torch.float)
        Y_tensor = torch.tensor(Y, dtype=torch.float).unsqueeze(-1)

        # Initialize the model
        model = SingleTaskGP(X_tensor, Y_tensor)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(100):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = -mll(output, Y_tensor)
            loss.backward()
            optimizer.step()

        best_f = Y.min().item()
        best_x = X[Y.argmin()]

        # Optimization loop
        for _ in range(self.budget - self.initial_points):
            # Define the acquisition function
            ei = ExpectedImprovement(model, best_f)

            # Optimize the acquisition function
            candidates, _ = optimize_acqf(
                acq_function=ei,
                bounds=self.bounds,
                q=1,
                num_restarts=5,
                raw_samples=20,
                options={"batch_limit": 5},
            )

            # Evaluate the new candidate
            new_x = candidates.numpy()
            new_x_denormalized = new_x * (ub.numpy() - lb.numpy()) + lb.numpy()
            new_y = func(new_x_denormalized)

            # Update the model
            X_tensor = torch.cat([X_tensor, candidates])
            Y_tensor = torch.cat([Y_tensor, torch.tensor(new_y, dtype=torch.float).unsqueeze(-1)])
            model = SingleTaskGP(X_tensor, Y_tensor)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            mll.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            for _ in range(100):
                optimizer.zero_grad()
                output = model(X_tensor)
                loss = -mll(output, Y_tensor)
                loss.backward()
                optimizer.step()

            # Update the best solution
            if new_y < best_f:
                best_f = new_y
                best_x = new_x_denormalized

        return best_f, best_x