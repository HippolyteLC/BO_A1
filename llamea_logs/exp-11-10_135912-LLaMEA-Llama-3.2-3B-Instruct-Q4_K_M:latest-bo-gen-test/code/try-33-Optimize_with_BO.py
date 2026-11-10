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
from torch.distributions import constraints
from torch.optim import L-BFGS-B
from torch.utils.data import Dataset, DataLoader
from scipy.special import roots_jacobi
import matplotlib.pyplot as plt
import pandas as pd

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed, acquisition_func=ExpectedImprovement, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.f_opt = np.inf
        self.x_opt = None
        self.model = Model(dim, acquisition_func=acquisition_func, likelihood=GaussianLikelihood(), device=device)

    def __call__(self, func):
        np.random.seed(self.random_seed)
        points = torch.randn(self.initial_points, self.dim)
        y = func(torch.randn(self.initial_points))

        self.model.fit(points, y)

        def acquisition_function(x):
            return self.model(x)

        for i in range(self.budget):
            x = torch.randn(1, self.dim).to(device)
            f = acquisition_function(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

        return self.f_opt, self.x_opt

    def optimize(self, func):
        self.__call__(func)
        return self.f_opt, self.x_opt

    def get_samples(self, func):
        self.__call__(func)
        points = torch.randn(self.initial_points, self.dim)
        y = func(points)
        return points, y

class Optimizer:
    def __init__(self, bo, func, budget, dim, initial_points, random_seed):
        self.bo = bo
        self.func = func
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def optimize(self):
        best_f, best_x = self.bo.optimize(self.func)
        return best_f, best_x

    def get_samples(self):
        points, y = self.bo.get_samples(self.func)
        return points, y

def main():
    # Define the function to optimize
    def func(x):
        return np.sum(x**2)  # a simple, multimodal function

    # Set up the Bayesian Optimization
    bo = Optimize_with_BO(budget=10000, dim=10, initial_points=10, random_seed=42, acquisition_func=ExpectedImprovement)

    # Perform the optimization
    optimizer = Optimizer(bo, func, 10000, 10, 10, 42)
    best_f, best_x = optimizer.optimize()
    print(f"Best function value: {best_f}")
    print(f"Best x: {best_x}")

    points, y = optimizer.get_samples()
    print(points)
    print(y)