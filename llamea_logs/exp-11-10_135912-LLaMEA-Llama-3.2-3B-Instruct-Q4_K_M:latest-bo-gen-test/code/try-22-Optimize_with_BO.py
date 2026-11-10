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

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed, acquisition_func=ExpectedImprovement, kernel_type='RBF'):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.f_opt = np.inf
        self.x_opt = None
        self.model = Model(dim, acquisition_func=acquisition_func, likelihood=GaussianLikelihood())
        self.kernel = self._create_kernel(kernel_type)

    def _create_kernel(self, kernel_type):
        if kernel_type == 'RBF':
            return torch.nn.functional.mssim
        elif kernel_type == 'Matern':
            return torch.nn.functional.matterhorn
        else:
            raise ValueError('Invalid kernel type')

    def __call__(self, func):
        np.random.seed(self.random_seed)
        points = torch.randn(self.initial_points, self.dim)
        y = func(points)

        self.model.fit(points, y)

        def acquisition_function(x):
            return self.model(x)

        for i in range(self.budget):
            x = torch.randn(1, self.dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            f = acquisition_function(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

            # Update the acquisition function
            x_acq = self.model(x)
            if i % 100 == 0:  # update every 100 iterations
                self.model.update_acquisition_function(x_acq)

        return self.f_opt, self.x_opt

def main():
    # Define the function to optimize
    def func(x):
        return np.sum(x**2)  # a simple, multimodal function

    # Set up the Bayesian Optimization
    bo = Optimize_with_BO(budget=10000, dim=10, initial_points=10, random_seed=42, acquisition_func=ExpectedImprovement, kernel_type='RBF')

    # Perform the optimization
    best_f, best_x = bo(func)

    print(f"Best function value: {best_f}")
    print(f"Best x: {best_x}")

# Update the acquisition function every 100 iterations
def update_acquisition_function(model, acquisition_function):
    model.update_acquisition_function(acquisition_function)

# Define the acquisition function to update
def update_acquisition_function_bo(model, acquisition_function):
    # Define a new acquisition function that is a combination of qLogExpectedImprovement and qUpperConfidenceBound
    def new_acquisition_function(x):
        f = acquisition_function(x)
        return qLogExpectedImprovement(f, model, acquisition_function, x) + qUpperConfidenceBound(f, model, acquisition_function, x)

    # Update the model with the new acquisition function
    model.update_acquisition_function(new_acquisition_function)