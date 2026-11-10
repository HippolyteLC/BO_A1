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

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed, acquisition_func=ExpectedImprovement, device=torch.device('cpu')):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.f_opt = np.inf
        self.x_opt = None
        self.model = Model(dim, acquisition_func=acquisition_func, likelihood=GaussianLikelihood())
        self.device = device

    def __call__(self, func):
        np.random.seed(self.random_seed)
        points = torch.randn(self.initial_points, self.dim)
        y = func(points)

        self.model.fit(points, y)

        def acquisition_function(x):
            return self.model(x).item()

        x_values = torch.linspace(-5, 5, self.dim).to(self.device).reshape(-1, self.dim)
        y_values = torch.randn(self.budget, self.dim).to(self.device)

        model_data = TensorDataset(points, y)
        model_dataloader = DataLoader(model_data, batch_size=1, shuffle=False)

        def get_acquisition_function(acquisition_func):
            def wrapper(x):
                with torch.no_grad():
                    output = acquisition_func(self.model, x, model_dataloader)
                return output.item()
            return wrapper

        acquisition_functions = {
            'ExpectedImprovement': get_acquisition_function(ExpectedImprovement),
            'UpperConfidenceBound': get_acquisition_function(UpperConfidenceBound),
            'LogExpectedImprovement': get_acquisition_function(qLogExpectedImprovement),
            'LogUpperConfidenceBound': get_acquisition_function(qLogUpperConfidenceBound),
            'SAWEI': get_acquisition_function(SAWEI)
        }

        for i in range(self.budget):
            for acquisition_func_name, acquisition_func in acquisition_functions.items():
                x = x_values[i % self.dim: (i % self.dim) + self.dim].reshape(1, self.dim)
                f = acquisition_func(acquisition_func_name, x, model_dataloader)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = x

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