import numpy as np
import torch
from torch import optim
from botorch.core import acquisition, kernels, models
from botorch.utils import get_hessian
import gpytorch
from gpytorch.models import ExactGPR
from gpytorch.acq import acquisition_function
from gpytorch.models import DeepGA
from botorch.optim import optimize_acq_function

class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int = 10, random_seed: int = 42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.model_gpr = ExactGPR(dim=self.dim)
        self.model_gpr.fit(torch.tensor(self.generate_initial_doe(), dtype=torch.float32), torch.tensor(self.generate_initial_y(), dtype=torch.float32), self.optimizer)
        self.acq_function = acquisition_function.QEI(self.model_gpr)
        self.optimizer = optim.Adam(self.model_gpr.parameters(), lr=0.01)
        self.x_doe = self.generate_initial_doe()

    def generate_initial_doe(self):
        np.random.seed(self.random_seed)
        x_doe = []
        for i in range(self.initial_points):
            x = np.random.uniform(-5, 5, self.dim)
            x_doe.append(x)
        return torch.tensor(x_doe, dtype=torch.float32)

    def generate_initial_y(self):
        np.random.seed(self.random_seed)
        y_doe = np.random.uniform(-5, 5, self.initial_points)
        return torch.tensor(y_doe, dtype=torch.float32)

    def __call__(self, func):
        for i in range(self.budget):
            x_doe = self.x_doe + torch.randn(self.initial_points, self.dim, device='cuda')
            self.x_doe = x_doe.detach()
            y_doe = func(torch.tensor(x_doe, dtype=torch.float32))

            self.model_gpr.fit(torch.tensor(x_doe), torch.tensor(y_doe), self.optimizer)

            z = self.acq_function(self.model_gpr, torch.tensor([0.0, 0.0], dtype=torch.float32))

            x = z + torch.randn(z.shape[0], self.dim, device='cuda')
            f = func(x)
            if f < self.model_gpr.f_min:
                self.model_gpr.f_min = f
                self.model_gpr.f_min_x = x

            self.x_doe = self.x_doe.detach()
        return self.model_gpr.f_min, self.model_gpr.f_min_x

def main():
    # Define the function to optimize
    def func(x):
        return np.sum(x**2)

    # Initialize the Optimize_with_BO class
    opt = Optimize_with_BO(budget=100, dim=5, initial_points=50)

    # Run the optimization
    best_f, best_x = opt(func)