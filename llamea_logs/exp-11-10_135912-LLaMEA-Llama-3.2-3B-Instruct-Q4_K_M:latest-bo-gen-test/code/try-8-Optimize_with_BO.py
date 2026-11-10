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
import ultraspob
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
        self.sobol_sampler = ultraspob.SobolSampler(random_seed=self.random_seed)
        self.dataloader = DataLoader(TensorDataset(torch.randn(self.initial_points, self.dim)), batch_size=1, shuffle=False)

    def __call__(self, func):
        points = torch.randn(self.initial_points, self.dim)
        y = func(points)
        self.model.fit(points, y)

        def acquisition_function(x):
            return self.model(x)

        for i, batch in enumerate(self.dataloader):
            x = batch
            f = acquisition_function(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

            # Add new points to the dataloader
            self.dataloader.add_batch(x, f)

        # Fit the model on the entire dataset
        self.model.fit(self.dataloader.dataset.x, self.dataloader.dataset.y)

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