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

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed, acquisition_func=ExpectedImprovement, device=torch.device('cpu')):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.f_opt = np.inf
        self.x_opt = None
        self.device = device
        self.model = Model(dim, acquisition_func=acquisition_func, likelihood=GaussianLikelihood()).to(device)

    def __call__(self, func):
        np.random.seed(self.random_seed)
        points = torch.randn(self.initial_points, self.dim)
        y = func(points)

        self.model.fit(points, y)

        def acquisition_function(x):
            return self.model(x)

        for i in range(self.budget):
            x = torch.randn(1, self.dim).to(self.device)
            f = acquisition_function(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

        return self.f_opt, self.x_opt

    def update_acq_func(self, acquisition_func):
        self.model.acquisition_function = acquisition_func

    def update_acq_model(self, model):
        self.model = model

def main():
    # Define the function to optimize
    def func(x):
        return np.sum(x**2)  # a simple, multimodal function

    # Set up the Bayesian Optimization
    bo = Optimize_with_BO(budget=10000, dim=10, initial_points=10, random_seed=42, acquisition_func=ExpectedImprovement)

    # Perform the optimization
    best_f, best_x = bo(func)

    print(f"Best function value: {best_f}")
    print(f"Best x: {best_x}")

    # Update the acquisition function to SAWEI
    bo.update_acq_func(SAWEI())

    # Update the acquisition model to a new one
    new_model = Model(dim, acquisition_func=SAWEI(), likelihood=GaussianLikelihood())
    bo.update_acq_model(new_model)

    # Perform the optimization again with the new acquisition function
    best_f, best_x = bo(func)

    print(f"Best function value with SAWEI: {best_f}")
    print(f"Best x with SAWEI: {best_x}")