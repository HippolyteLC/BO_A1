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
from torch.distributions import Normal
from scipy.special import ellipkiv
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

    def __call__(self, func):
        np.random.seed(self.random_seed)
        points = torch.randn(self.initial_points, self.dim).to(self.device)
        y = torch.tensor(func(points.detach().numpy())).to(self.device)
        self.model.fit(points, y)

        def acquisition_function(x):
            return self.model(x).mean

        for i in range(self.budget):
            x = torch.randn(1, self.dim).to(self.device)
            f = acquisition_function(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

        return self.f_opt, self.x_opt

    def plot_acquisition(self, acquisition_func):
        x = torch.linspace(-5, 5, 1000).to(self.device)
        f = acquisition_func(x)
        plt.plot(x.numpy(), f.numpy())
        plt.show()

    def plot_posterior(self):
        x = torch.linspace(-5, 5, 1000).to(self.device)
        posterior = self.model(x)
        plt.plot(x.numpy(), posterior.numpy())
        plt.show()

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

    # Plot the acquisition function
    bo.plot_acquisition(lambda x: bo.model(x))

    # Plot the posterior distribution
    bo.plot_posterior()