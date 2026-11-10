import numpy as np
import torch
from torch import optim
from botorch.core import acquisition, kernels, models, priors, utilities
from botorch.utils import get_hessian
from gpytorch.models import ExactGP, MultitaskGP
from gpytorch.priors import ConstantPrior
from gpytorchAcq import qExpectedImprovement, qLogExpectedImprovement
from torch import nn, Tensor
from torch.nn import functional as F

class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int = 10, random_seed: int = 42, kernel_type: str = 'Matern32'):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.model = ExactGP(dim=self.dim, likelihood=F.gaussian_likelihood)
        if kernel_type == 'Matern32':
            self.model.kernels = kernels.Matern32Kernel()
        elif kernel_type == 'RBF':
            self.model.kernels = kernels.RBFKernel()
        elif kernel_type == 'Scale':
            self.model.kernels = kernels.ScaleKernel()
        elif kernel_type == 'Matern':
            self.model.kernels = kernels.MaternKernel()
        self.acq = qExpectedImprovement(self.model, self.model.likelihood)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.x_doe = self.generate_initial_doe()
        self.f_min = np.inf
        self.f_min_x = None

    def generate_initial_doe(self):
        np.random.seed(self.random_seed)
        x_doe = []
        for i in range(self.initial_points):
            x = np.random.uniform(-5, 5, self.dim)
            x_doe.append(x)
        return torch.tensor(x_doe, dtype=torch.float32)

    def __call__(self, func):
        for i in range(self.budget):
            x_doe = self.x_doe + torch.randn(self.initial_points, self.dim, device='cuda')
            self.x_doe = x_doe.detach()
            y_doe = func(torch.tensor(x_doe, dtype=torch.float32))

            self.model.fit(self.x_doe, y_doe, self.optimizer)

            z = self.model.predict(torch.tensor(self.x_doe, dtype=torch.float32), self.model.likelihood)
            best_x = self.acq(z, torch.tensor([0.0, 0.0], dtype=torch.float32))

            x = best_x + torch.randn(best_x.shape[0], self.dim, device='cuda')
            f = func(x)
            if f < self.f_min:
                self.f_min = f
                self.f_min_x = x

            self.x_doe = self.x_doe.detach()
        return self.f_min, self.f_min_x