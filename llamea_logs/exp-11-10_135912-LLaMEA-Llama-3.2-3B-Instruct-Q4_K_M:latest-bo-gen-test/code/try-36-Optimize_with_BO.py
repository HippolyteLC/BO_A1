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
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import qmc

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed, acquisition_func=ExpectedImprovement, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
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
        y = func(torch.randn(self.initial_points))
        
        # Fit the GPyTorch model
        self.model.fit(points, y)

        # Define the acquisition function
        def acquisition_function(x):
            return self.model(x).item()

        # Define the acquisition function for the SAWEI acquisition
        def sawei_acquisition_function(x):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            qf = qmc.qmc(x, self.model, sampler='quasi-Monte')
            return qf.mean().item()

        # Perform Bayesian Optimization
        for i in range(self.budget):
            # Randomly sample a point
            x = torch.randn(1, self.dim).to(device=self.device)

            # Fit the model at the new point
            f = acquisition_function(x)
            if f < self.f_opt:
                # Update the best point
                self.f_opt = f
                self.x_opt = x

        # Perform SAWEI acquisition function
        sawei_x = np.linspace(self.x_opt[0], self.x_opt[0] + self.budget * 0.1, self.budget).reshape(-1, 1)
        sawei_y = [acquisition_function(torch.tensor(x, dtype=torch.float32, device=self.device)) for x in sawei_x]
        sawei_points = torch.tensor(swei_x, dtype=torch.float32, device=self.device)
        sawei_dataset = TensorDataset(swei_points, torch.tensor(swei_y, dtype=torch.float32, device=self.device))
        sawei_dataloader = DataLoader(swei_dataset, batch_size=32, shuffle=False)

        # Fit the GPyTorch model for SAWEI
        with torch.no_grad():
            for batch in sawei_dataloader:
                batch_x, batch_y = batch
                batch_x = batch_x.to(device=self.device)
                batch_y = batch_y.to(device=self.device)
                self.model(batch_x, batch_y)

        # Evaluate SAWEI acquisition function
        for i, x in enumerate(swei_x):
            f = sawei_acquisition_function(x)
            if f < self.f_opt:
                # Update the best point
                self.f_opt = f
                self.x_opt = x

        return self.f_opt, self.x_opt

def main():
    # Define the function to optimize
    def func(x):
        return np.sum(x**2)  # a simple, multimodal function

    # Set up the Bayesian Optimization
    bo = Optimize_with_BO(budget=10000, dim=10, initial_points=10, random_seed=42, acquisition_func=ExpectedImprovement, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    # Perform the optimization
    best_f, best_x = bo(func)

    print(f"Best function value: {best_f}")
    print(f"Best x: {best_x}")