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
from torch.optim import Adam
from torch.distributions import Normal

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed, acquisition_func=ExpectedImprovement):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.f_opt = np.inf
        self.x_opt = None
        self.model = Model(dim, acquisition_func=acquisition_func, likelihood=GaussianLikelihood())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device()

    def __call__(self, func):
        np.random.seed(self.random_seed)
        points = torch.randn(self.initial_points, self.dim)
        y = func(torch.randn(self.initial_points))

        self.model.fit(points, y)
        self.model.to(self.device)

        def acquisition_function(x):
            return self.model(x).mean

        for i in range(self.budget):
            x = torch.randn(1, self.dim).to(self.device)
            f = acquisition_function(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

            # Update the acquisition function
            if i < self.budget - 1:
                # Define a custom acquisition function using qLogExpectedImprovement and qUpperConfidenceBound
                custom_acq_func = qLogExpectedImprovement(self.model, self.model.likelihood, acquisition_func=self.model.acquisition_function) + qUpperConfidenceBound(self.model, self.model.likelihood, self.model.acquisition_function)
                # Update the acquisition function
                self.model.acquisition_function = custom_acq_func

        return self.f_opt, self.x_opt

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

    # Update the model to use the custom acquisition function
    def custom_acq_func(x):
        # Get the predictions from the model
        mean, std = self.model(x).mean, self.model(x).std
        # Define the custom acquisition function
        return qLogExpectedImprovement(self.model, self.model.likelihood, acquisition_func=self.model.acquisition_function) + qUpperConfidenceBound(self.model, self.model.likelihood, self.model.acquisition_function)
    bo.model.acquisition_function = custom_acq_func

    # Perform another iteration of the optimization
    best_f, best_x = bo(func)

    print(f"Best function value (updated): {best_f}")
    print(f"Best x (updated): {best_x}")