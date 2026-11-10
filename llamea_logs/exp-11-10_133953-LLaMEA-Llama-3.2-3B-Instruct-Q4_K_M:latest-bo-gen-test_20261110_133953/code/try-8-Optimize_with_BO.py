import numpy as np
import torch
import gpytorch
from botorch.core import acquisition, kernels, models
from botorch.utils import get_hessian
from scipy.stats import qmc

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
            # Use Latin Hypercube Sampling to evaluate the function
            lhs = qmc.LatinHypercube(dim=self.dim)
            lhs_points = lhs.random(self.initial_points)
            lhs_points = torch.tensor(lhs_points, dtype=torch.float32)
            y_lhs = func(lhs_points)
            # Update the model with the new data
            self.model.update_data(self.x_doe, y_doe)
            self.model.update_data(lhs_points, y_lhs)

        return self.model.f_min, self.model.f_min_x