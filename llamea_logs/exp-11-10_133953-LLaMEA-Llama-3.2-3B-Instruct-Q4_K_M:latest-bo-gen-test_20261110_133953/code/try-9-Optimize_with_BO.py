import numpy as np
import torch
from torch import optim
from botorch.core import acquisition, kernels, models
from botorch.utils import get_hessian
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from gpytorch.models import ExactGP, MultitaskGP
from gpytorch.priors import Scale
from botorch import Matern, ScaleKernel, qUpperConfidenceBound, qSAWEI

class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int = 10, random_seed: int = 42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.model = ExactGP(dim=self.dim)
        self.acq = qExpectedImprovement(self.model, self.model.acq_function)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.x_doe = self.generate_initial_doe()
        self.kernels = ScaleKernel(
            Matern(dim=self.dim, length_scale=1.0),
            Scale(dim=self.dim, variance=1.0)
        )
        self.model.kernels.set_kernels(self.kernels)
        self.aux_acq_function = qLogExpectedImprovement(self.model, self.model.acq_function)

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

            self.model.fit(y_doe, x_doe, self.optimizer)

            z = self.model.predict(torch.tensor(x_doe, dtype=torch.float32), self.model.acq_function)
            best_x = self.acq(z, torch.tensor([0.0, 0.0], dtype=torch.float32))

            x = best_x + torch.randn(best_x.shape[0], self.dim, device='cuda')
            f = func(x)
            if f < self.model.f_min:
                self.model.f_min = f
                self.model.f_min_x = x

            self.x_doe = self.x_doe.detach()
        return self.model.f_min, self.model.f_min_x

    def update_strategy(self, new_strategy):
        if new_strategy == 'qSAWEI':
            self.acq = qSAWEI(self.model, self.model.acq_function, self.aux_acq_function)
        elif new_strategy == 'qUCB':
            self.acq = qUpperConfidenceBound(self.model, self.model.acq_function, self.aux_acq_function)
        else:
            self.acq = qExpectedImprovement(self.model, self.model.acq_function)