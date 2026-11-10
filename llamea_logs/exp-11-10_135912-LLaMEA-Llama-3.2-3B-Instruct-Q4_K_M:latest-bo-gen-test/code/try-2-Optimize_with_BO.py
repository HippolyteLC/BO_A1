import numpy as np
import torch
from torch import tensor
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound, qLogExpectedImprovement, qLogUpperConfidenceBound
from botorch.models import Model, SingleTaskModel
from botorch.acquisition import SAWEI
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import RMSprop
from bo_torch.acquisitions import qLogSAWEI

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed, acquisition_func=ExpectedImprovement, optim_method=Adam):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.f_opt = np.inf
        self.x_opt = None
        self.model = Model(dim, acquisition_func=acquisition_func, likelihood=GaussianLikelihood())
        self.optim_method = optim_method

    def __call__(self, func):
        np.random.seed(self.random_seed)
        points = torch.randn(self.initial_points, self.dim)
        y = func(torch.randn(self.initial_points))

        self.model.fit(points, y)

        def acquisition_function(x):
            return self.model(x)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_tensor_type('torch.DoubleTensor')
        torch.set_default_tensor_type('torch.DoubleTensor')

        x_train = points.to(device)
        y_train = y.to(device)
        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        best_f, best_x = self.optimize(loader, acquisition_function, acquisition_func)
        return best_f, best_x

    def optimize(self, loader, acquisition_function, acquisition_func):
        for i in range(self.budget):
            x = torch.randn(1, self.dim).to(device)
            f = acquisition_function(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

        return self.f_opt, self.x_opt

def main():
    # Define the function to optimize
    def func(x):
        return np.sum(x**2)  # a simple, multimodal function

    # Set up the Bayesian Optimization
    bo = Optimize_with_BO(budget=10000, dim=10, initial_points=10, random_seed=42, acquisition_func=ExpectedImprovement, optim_method=Adam)

    # Perform the optimization
    best_f, best_x = bo(func)

    print(f"Best function value: {best_f}")
    print(f"Best x: {best_x}")