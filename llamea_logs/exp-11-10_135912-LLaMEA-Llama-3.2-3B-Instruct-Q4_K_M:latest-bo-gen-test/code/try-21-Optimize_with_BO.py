import numpy as np
import torch
from torch import tensor
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound, qLogExpectedImprovement, qLogUpperConfidenceBound
from botorch.models import Model, SingleTaskModel
from botorch.acquisition import SAWEI
from torch import nn
from botorch.utils import get_device
from gpytorch.kernels import MaternKernel, ScaleKernel
from torch.optim import Adam

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed, acquisition_func=ExpectedImprovement, kernel=MaternKernel()):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.f_opt = np.inf
        self.x_opt = None
        self.model = Model(dim, acquisition_func=acquisition_func, likelihood=GaussianLikelihood())
        self.device = get_device()
        self.kernel = kernel.to(self.device)

    def __call__(self, func):
        np.random.seed(self.random_seed)
        points = torch.randn(self.initial_points, self.dim)
        y = func(torch.randn(self.initial_points))

        self.model.fit(points, y, self.kernel)

        def acquisition_function(x):
            return self.model(x)

        for i in range(self.budget):
            x = torch.randn(1, self.dim).to(self.device)
            f = acquisition_function(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

            # update the model for the next iteration
            self.model.fit(points, torch.cat((points, torch.tensor([[x]])), dim=0), y, self.kernel)
            self.model.parameters = [p for p in self.model.parameters if p.requires_grad]

        # use the last model for the last iteration
        self.model = Model(self.dim, acquisition_func=self.acquisition_func, likelihood=GaussianLikelihood())
        self.model.fit(points, y, self.kernel)

        return self.f_opt, self.x_opt

    def acquisition_func(self, x):
        return self.model(x)

# Example usage
def func(x):
    return np.sum(x**2)  # a simple, multimodal function

bo = Optimize_with_BO(budget=10000, dim=10, initial_points=10, random_seed=42, acquisition_func=ExpectedImprovement, kernel=ScaleKernel())

best_f, best_x = bo(func)

print(f"Best function value: {best_f}")
print(f"Best x: {best_x}")