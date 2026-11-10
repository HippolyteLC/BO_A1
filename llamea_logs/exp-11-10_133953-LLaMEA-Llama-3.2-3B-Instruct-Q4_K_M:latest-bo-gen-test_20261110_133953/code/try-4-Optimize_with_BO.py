import numpy as np
import torch
from torch import optim
from botorch.core import acquisition, kernels, models
from botorch.utils import get_hessian
from botorch.acquisition import qNoisyExpectedImprovement
from gpytorch.models import ExactGP
from gpytorch.acquisition import Acquisition
from scipy.optimize import minimize
from scipy.stats import norm

class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int = 10, random_seed: int = 42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.model = ExactGP(dim=self.dim, likelihood=norm).to(torch.device('cuda'))
        self.acq = qNoisyExpectedImprovement(self.model, self.model.acq_function)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.x_doe = self.generate_initial_doe()
        self.n_iter = 0

    def generate_initial_doe(self):
        np.random.seed(self.random_seed)
        x_doe = []
        for i in range(self.initial_points):
            x = np.random.uniform(-5, 5, self.dim)
            x_doe.append(x)
        return torch.tensor(x_doe, dtype=torch.float32)

    def __call__(self, func):
        for i in range(self.budget):
            self.n_iter += 1
            x_doe = self.x_doe + torch.randn(self.initial_points, self.dim, device='cuda')
            self.x_doe = x_doe.detach()
            y_doe = func(torch.tensor(x_doe, dtype=torch.float32))

            self.model.fit(self.x_doe, y_doe, self.optimizer)

            z = self.model.predict(torch.tensor(self.x_doe, dtype=torch.float32), self.model.acq_function)
            best_x = self.acq(z, torch.tensor([0.0, 0.0], dtype=torch.float32))

            x = best_x + torch.randn(best_x.shape[0], self.dim, device='cuda')
            f = func(x)
            if f < self.model.f_min:
                self.model.f_min = f
                self.model.f_min_x = x

            self.x_doe = self.x_doe.detach()

        # Update acquisition function
        self.acq = qNoisyExpectedImprovement(self.model, self.model.acq_function)

        # Perform optimization
        def negacq_func(x):
            z = self.model.predict(torch.tensor(x, dtype=torch.float32), self.model.acq_function)
            return -self.acq(z, torch.tensor([0.0, 0.0], dtype=torch.float32)).item()

        res = minimize(negacq_func, torch.tensor([0.0] * self.dim, dtype=torch.float32), method="SLSQP", options={"maxiter": 1000})
        self.x_doe = res.x

        return self.model.f_min, self.model.f_min_x