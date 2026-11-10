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
from torch.utils.data import TensorDataset, DataLoader

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
        self.model.to(self.device)

    def __call__(self, func):
        np.random.seed(self.random_seed)
        points = torch.randn(self.initial_points, self.dim)
        y = func(torch.randn(self.initial_points))

        self.model.fit(points, y)

        def acquisition_function(x):
            return self.model(x)

        for i in range(self.budget):
            x = torch.randn(1, self.dim).to(self.device)
            f = acquisition_function(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

            # Use acquisition function to update the model
            self.model.update(x, f)

        # Perform Bayesian Optimization
        data = TensorDataset(points, y)
        loader = DataLoader(data, batch_size=1, shuffle=False)

        # Define the acquisition function
        if acquisition_func == ExpectedImprovement:
            acquisition_func_ = ExpectedImprovement(self.model, self.model.likelihood)
        elif acquisition_func == UpperConfidenceBound:
            acquisition_func_ = UpperConfidenceBound(self.model, self.model.likelihood)
        elif acquisition_func == SAWEI:
            acquisition_func_ = SAWEI(self.model, self.model.likelihood)
        else:
            raise ValueError("Invalid acquisition function")

        # Perform Bayesian Optimization
        x_new = torch.randn(1, self.dim).to(self.device)
        f_new = acquisition_func_(x_new)
        self.model.update(x_new, f_new)

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

    # Refine the strategy by using a different acquisition function
    #bo = Optimize_with_BO(budget=10000, dim=10, initial_points=10, random_seed=42, acquisition_func=SAWEI)

    # Perform the optimization
    best_f, best_x = bo(func)

    print(f"Best function value: {best_f}")
    print(f"Best x: {best_x}")