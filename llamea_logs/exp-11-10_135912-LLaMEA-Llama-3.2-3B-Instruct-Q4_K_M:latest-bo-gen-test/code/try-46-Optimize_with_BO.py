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
import torch.nn.functional as F
from scipy.stats import norm

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed, acquisition_func=ExpectedImprovement):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.f_opt = np.inf
        self.x_opt = None
        self.model = Model(dim, acquisition_func=acquisition_func, likelihood=GaussianLikelihood())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optim = Adam(self.model.parameters(), lr=0.001)

    def __call__(self, func):
        np.random.seed(self.random_seed)
        points = torch.randn(self.initial_points, self.dim)
        y = func(torch.randn(self.initial_points))

        self.model.fit(points, y)

        def acquisition_function(x):
            return self.model(x)

        # Select acquisition function
        if acquisition_func == ExpectedImprovement:
            acq_func = ExpectedImprovement(acquisition_function)
        elif acquisition_func == UpperConfidenceBound:
            acq_func = UpperConfidenceBound(acquisition_function)
        elif acquisition_func == ProbabilityOfImprovement:
            acq_func = ProbabilityOfImprovement(acquisition_function)
        elif acquisition_func == qExpectedImprovement:
            acq_func = qExpectedImprovement(acquisition_function)
        elif acquisition_func == qUpperConfidenceBound:
            acq_func = qUpperConfidenceBound(acquisition_function)
        elif acquisition_func == SAWEI:
            acq_func = SAWEI(acquisition_function)

        # Perform Bayesian Optimization
        best_x = None
        best_f = np.inf
        for i in range(self.budget):
            x = torch.randn(1, self.dim).to(self.device)
            f = acquisition_function(x)
            if f < best_f:
                best_x = x
                best_f = f

            # Update model and acquisition function
            self.optim.zero_grad()
            acq_func.loss.backward()
            self.optim.step()

            # If best f is found, stop optimization
            if i == self.budget - 1:
                break

        return best_f, best_x

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

# Refine the strategy of the selected solution to improve it
# One approach is to use a combination of acquisition functions, such as the SAWEI
# Another approach is to use a more advanced acquisition function, such as the qLogExpectedImprovement
# or the qUpperConfidenceBound with a more complex kernel, such as the Matern kernel