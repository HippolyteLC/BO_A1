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
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import rubi
import ultrfastgp
from gpytorch.likelihoods import GaussianLikelihood
from torch.distributions import MultivariateNormal

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
        points = torch.randn(self.initial_points, self.dim).to(device)
        y = func(torch.randn(self.initial_points)).to(device)
        
        self.model.fit(points, y)

        def acquisition_function(x):
            return self.model(x)

        # Generate new points using the surrogate
        new_points = self.model.surrogate
        new_points = new_points.detach().numpy().T
        
        # Compute the upper confidence bound
        upper_bounds = qUpperConfidenceBound(acquisition_function, new_points, device=device)
        upper_bounds = torch.as_tensor(upper_bounds).numpy()
        
        # Compute the expected improvement
        exp_improvement = qExpectedImprovement(acquisition_function, new_points, device=device)
        exp_improvement = torch.as_tensor(exp_improvement).numpy()
        
        # Compute the lower confidence bound
        lower_bounds = qLowerConfidenceBound(acquisition_function, new_points, device=device)
        lower_bounds = torch.as_tensor(lower_bounds).numpy()
        
        # Use rubi to select points
        selected_points = rubi(new_points, upper_bounds, lower_bounds)
        selected_points = torch.as_tensor(selected_points).numpy()
        
        # Select points with the highest expected improvement
        selected_points = torch.tensor(selected_points[np.argsort(exp_improvement)[::-1]])
        
        # Remove duplicate points
        selected_points = torch.unique(selected_points, axis=0)
        
        # Select points with the highest confidence
        selected_points = torch.tensor(selected_points[np.argsort(upper_bounds)[::-1]])
        
        # Add the new points to the points to evaluate
        points = torch.cat((points, selected_points), axis=0)
        
        # Compute the new function values
        y = func(points)
        
        # Update the model
        self.model.fit(points, y)
        
        # Get the best function value and point
        if y.min() < self.f_opt:
            self.f_opt = y.min()
            self.x_opt = points[-1]
        
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