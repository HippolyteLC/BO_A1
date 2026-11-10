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
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.mlls import ExactMarginalLogLikelihood

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed, acquisition_func=ExpectedImprovement):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.f_opt = np.inf
        self.x_opt = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = Model(dim, acquisition_func=acquisition_func, likelihood=GaussianLikelihood()).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    def __call__(self, func):
        np.random.seed(self.random_seed)
        points = torch.randn(self.initial_points, self.dim)
        y = func(torch.randn(self.initial_points))
        y = y.to(self.device)

        dataset = TensorDataset(points, y)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        for i in range(self.budget):
            batch_points, batch_y = next(iter(dataloader))
            batch_points = batch_points.to(self.device)
            self.optimizer.zero_grad()
            self.mll.batch_fit(batch_points, batch_y)
            self.optimizer.step()

            def acquisition_function(x):
                return self.model(x)

            x = torch.randn(1, self.dim).to(self.device)
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
    bo = Optimize_with_BO(budget=10000, dim=10, initial_points=10, random_seed=42, acquisition_func=SAWEI)

    # Perform the optimization
    best_f, best_x = bo(func)

    print(f"Best function value: {best_f}")
    print(f"Best x: {best_x}")
