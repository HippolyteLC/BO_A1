import numpy as np
import torch
import torch.nn.functional as F
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound, qLogExpectedImprovement, qLogUpperConfidenceBound
from botorch.models import Model, SingleTaskModel
from botorch.acquisition import SAWEI
from torch import nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed, acquisition_func=ExpectedImprovement, kernel_type='RBF', likelihood_type='Gaussian'):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.f_opt = np.inf
        self.x_opt = None
        self.model = Model(dim, acquisition_func=acquisition_func, likelihood=GaussianLikelihood(), kernel_type=kernel_type, likelihood_type=likelihood_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def __call__(self, func):
        np.random.seed(self.random_seed)
        points = torch.randn(self.initial_points, self.dim).to(self.device)
        y = func(torch.randn(self.initial_points)).to(self.device)

        self.model.fit(points, y)

        def acquisition_function(x):
            return self.model(x)

        for i in range(self.budget):
            x = torch.randn(1, self.dim).to(self.device)
            f = acquisition_function(x).item()
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

        return self.f_opt, self.x_opt

def main():
    # Define the function to optimize
    def func(x):
        return np.sum(x**2)  # a simple, multimodal function

    # Set up the Bayesian Optimization
    bo = Optimize_with_BO(budget=10000, dim=10, initial_points=10, random_seed=42, acquisition_func=ExpectedImprovement, kernel_type='RBF', likelihood_type='Gaussian')

    # Perform the optimization
    best_f, best_x = bo(func)

    print(f"Best function value: {best_f}")
    print(f"Best x: {best_x}")

    # Refine the strategy of the selected solution to improve it
    def refine_acquisition(x):
        # Use SAWEI acquisition function
        sawei = SAWEI(acquisition_func=ExpectedImprovement, kernel_type='RBF', likelihood_type='Gaussian')
        return sawei(x).item()

    # Use the refined acquisition function to get the next point
    x_next = torch.randn(1, self.dim).to(self.device)
    f_next = refine_acquisition(x_next)
    if f_next < self.f_opt:
        self.f_opt = f_next
        self.x_opt = x_next

    print(f"Refined best function value: {self.f_opt}")
    print(f"Refined best x: {self.x_opt}")