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
from torch.utils.data import Dataset, DataLoader
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

    def __call__(self, func):
        np.random.seed(self.random_seed)
        points = torch.randn(self.initial_points, self.dim)
        y = func(torch.randn(self.initial_points))

        self.model.fit(points, y)

        device = points.device
        acquisition_function = qExpectedImprovement(self.model, acq_optimizer=nn.Adam(self.model.parameters(), lr=0.01))
        for i in range(self.budget):
            x = torch.randn(1, self.dim).to(device)
            f = acquisition_function(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

        return self.f_opt, self.x_opt

class CustomDataset(Dataset):
    def __init__(self, points, y):
        self.points = points
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.points[idx], self.y[idx]

def main():
    # Define the function to optimize
    def func(x):
        return np.sum(x**2)  # a simple, multimodal function

    # Set up the Bayesian Optimization
    bo = Optimize_with_BO(budget=10000, dim=10, initial_points=10, random_seed=42, acquisition_func=ExpectedImprovement)

    # Define the dataset
    dataset = CustomDataset(points=torch.randn(10, 10), y=func(torch.randn(10)))

    # Create the data loader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Perform the optimization
    for epoch in range(10):
        for batch in data_loader:
            x, y = batch
            f = bo(x)[0]
            if f < bo.f_opt:
                bo.f_opt = f
                bo.x_opt = x
        print(f"Epoch {epoch+1}, Best function value: {bo.f_opt}")

    print(f"Best function value: {bo.f_opt}")
    print(f"Best x: {bo.x_opt}")
