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
    def __init__(self, budget, dim, initial_points, random_seed, acquisition_func=ExpectedImprovement):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.f_opt = np.inf
        self.x_opt = None
        self.model = Model(dim, acquisition_func=acquisition_func, likelihood=GaussianLikelihood())

    def __call__(self, func):
        np.random.seed(self.random_seed)
        points = torch.randn(self.initial_points, self.dim)
        y = func(torch.randn(self.initial_points))

        self.model.fit(points, y)

        def acquisition_function(x):
            return self.model(x)

        # Define a custom device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for i in range(self.budget):
            # Generate random points
            x = torch.randn(1, self.dim).to(device)
            f = acquisition_function(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

        # Return the optimized values
        return self.f_opt, self.x_opt

def main():
    # Define the function to optimize
    def func(x):
        return np.sum(x**2)  # a simple, multimodal function

    # Set up the Bayesian Optimization
    bo = Optimize_with_BO(budget=10000, dim=10, initial_points=10, random_seed=42, acquisition_func=ExpectedImprovement)

    # Define a custom device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Perform the optimization
    best_f, best_x = bo(func)

    print(f"Best function value: {best_f}")
    print(f"Best x: {best_x}")

# Update the model to use the best points
def update_model(bo, func, new_points, new_y):
    # Update the model with the new points and values
    points = torch.cat((bo.model.y_train.values(), torch.tensor(new_y.values()).to(device)))
    y = torch.cat((bo.model.y_train.names(), torch.tensor(new_y.names()).to(device)))

    bo.model.fit(points, y)

# Update the model to use the best points and new function values
def update_model_new_points(bo, func, new_points, new_y):
    # Update the model with the new points and values
    points = torch.cat((bo.model.y_train.values(), torch.tensor(new_y.values()).to(device)))
    y = torch.cat((bo.model.y_train.names(), torch.tensor(new_y.names()).to(device)))

    # Define a custom device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bo.model.fit(points, y)

    # Return the optimized values
    return points, y

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

    # Update the model to use the best points
    def new_func(x):
        return np.sum(x**2 + 0.1 * np.sin(x))  # a simple, multimodal function

    new_points = torch.randn(10, 10)
    new_y = new_func(new_points)

    update_model(bo, new_func, new_points, new_y)

    # Perform the optimization
    best_f, best_x = bo(func)

    print(f"Best function value: {best_f}")
    print(f"Best x: {best_x}")

    # Update the model to use the best points and new function values
    def new_func(x):
        return np.sum(x**2 + 0.1 * np.sin(x))  # a simple, multimodal function

    new_points = torch.randn(10, 10)
    new_y = new_func(new_points)

    update_model_new_points(bo, new_func, new_points, new_y)

    # Perform the optimization
    best_f, best_x = bo(func)

    print(f"Best function value: {best_f}")
    print(f"Best x: {best_x}")