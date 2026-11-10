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
from scipy.stats import sobol
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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
        self.sobol_set = sobol.dims(dim, r=initial_points)
        self.f_values = np.zeros(initial_points)
        self.x_values = np.zeros((initial_points, dim))
        self.x_values[:, 0] = np.random.uniform(-5, 5, size=dim)
        for i in range(initial_points):
            self.f_values[i] = func(self.x_values[i])
        self.dataset = TensorDataset(torch.tensor(self.x_values), torch.tensor(self.f_values))
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)

    def __call__(self, func):
        # Fit the model using the initial points
        for i, (x, y) in enumerate(self.data_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            self.model.fit(x, y)

            # Get the acquisition function values
            with torch.no_grad():
                x = torch.randn(1, self.dim).to(self.device)
                f = acquisition_function(x)
                self.f_opt = min(self.f_opt, f.item())
                self.x_opt = x.item()

        return self.f_opt, self.x_opt

def acquisition_function(x):
    # Get the posterior mean
    with torch.no_grad():
        posterior_mean = self.model.posterior_mean(x)
        # Get the posterior variance
        posterior_var = self.model.posterior_var(x)
        # Get the acquisition function value
        f = qExpectedImprovement(acquisition_function=self.model, X=x).to(self.device)
        f = f - posterior_mean * torch.sqrt(posterior_var)
    return f

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

# Additional code to plot the initial DoE points
    f_values = bo.f_values
    x_values = bo.x_values
    plt.scatter(x_values[:, 0], f_values)
    plt.show()