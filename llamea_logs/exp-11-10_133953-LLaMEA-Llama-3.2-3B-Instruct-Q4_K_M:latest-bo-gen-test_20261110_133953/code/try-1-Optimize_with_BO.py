import numpy as np
import torch
from torch import optim
from botorch.core import acquisition, kernels, models
from botorch.utils import get_hessian
from torch.nn import functional as F

class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int = 10, random_seed: int = 42):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.model = models.GPR(dim=self.dim)
        self.acq = acquisition.QEI(self.model, self.model.acq_function)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.x_doe = self.generate_initial_doe()
        self.y_doe = []
        self.f_min = np.inf
        self.f_min_x = None

    def generate_initial_doe(self):
        np.random.seed(self.random_seed)
        x_doe = []
        for i in range(self.initial_points):
            x = np.random.uniform(-5, 5, self.dim)
            x_doe.append(x)
        return torch.tensor(x_doe, dtype=torch.float32)

    def fit_model(self, x_doe, y_doe):
        self.model.fit(x_doe, y_doe, self.optimizer)

    def predict(self, x_doe):
        return self.model.predict(x_doe, self.model.aux_acq_function)

    def acq_function(self, z):
        return self.acq(z, torch.tensor([0.0, 0.0], dtype=torch.float32))

    def update_x_doe(self, x_doe):
        self.x_doe = x_doe.detach()

    def update_y_doe(self, y_doe):
        self.y_doe = y_doe.detach()

    def update_f_min(self, f_min):
        self.f_min = f_min

    def update_f_min_x(self, f_min_x):
        self.f_min_x = f_min_x

    def __call__(self, func):
        for i in range(self.budget):
            x_doe = self.x_doe + torch.randn(self.initial_points, self.dim, device='cuda')
            self.update_x_doe(x_doe)
            y_doe = func(torch.tensor(x_doe, dtype=torch.float32))
            self.update_y_doe(y_doe)

            self.fit_model(x_doe, y_doe)

            z = self.predict(x_doe)
            best_x = self.acq_function(z)
            x = best_x + torch.randn(best_x.shape[0], self.dim, device='cuda')

            f = func(x)
            if f < self.f_min:
                self.update_f_min(f)
                self.update_f_min_x(x)

            self.update_x_doe(x_doe)
        return self.f_min, self.f_min_x