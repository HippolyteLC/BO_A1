import numpy as np
import torch
from botorch.core import acquisition, kernels, models
from botorch.utils import get_hessian
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from gpytorch.models import ExactGP
from gpytorch.acq import qExpectedImprovement, qLogExpectedImprovement
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int = 10, random_seed: int = 42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.model = ExactGP(dim=self.dim)
        self.acq = qLogExpectedImprovement(self.model)
        self.optimizer = Adam(self.model.parameters(), lr=0.01)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.2, patience=5, min_lr=0.001)
        self.x_doe = self.generate_initial_doe()

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

            z = self.model.predict(torch.tensor(self.x_doe, dtype=torch.float32), self.model.aux_acq_function)
            best_x = self.acq(z, torch.tensor([0.0, 0.0], dtype=torch.float32))

            x = best_x + torch.randn(best_x.shape[0], self.dim, device='cuda')
            f = func(x)
            if f < self.model.f_min:
                self.model.f_min = f
                self.model.f_min_x = x

            self.x_doe = self.x_doe.detach()
            self.optimizer = self.optimizer
            self.lr_scheduler = self.lr_scheduler
        return self.model.f_min, self.model.f_min_x

    def update_model(self, func):
        self.model = ExactGP(dim=self.dim)
        self.acq = qLogExpectedImprovement(self.model)
        self.optimizer = Adam(self.model.parameters(), lr=0.01)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.2, patience=5, min_lr=0.001)

    def evaluate_function(self, func, x):
        y = func(x)
        return y

    def get_acq_function(self):
        return self.acq