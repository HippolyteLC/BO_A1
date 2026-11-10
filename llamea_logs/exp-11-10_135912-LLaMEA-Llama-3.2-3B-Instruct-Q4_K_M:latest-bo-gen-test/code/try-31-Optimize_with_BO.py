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
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import minimize
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

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)

        acquisition_function = self.model

        for i in range(self.budget):
            x = torch.randn(1, self.dim).to(device)
            f = acquisition_function(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

        return self.f_opt, self.x_opt

class CustomDataset(Dataset):
    def __init__(self, x, y, acquisition_func):
        self.x = torch.tensor(x, dtype=torch.float32).view(-1, self.dim)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.acquisition_func = acquisition_func

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        f = self.acquisition_func(x)
        return x, y, f

def custom_acquisition_function(X, y, acquisition_func, model):
    dataset = CustomDataset(X, y, acquisition_func)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for X_batch, y_batch, f_batch in dataloader:
            X_batch = model(X_batch)
            f_batch = acquisition_func(X_batch)
            f_batch = torch.mean(f_batch, dim=0)
            return f_batch

def main():
    # Define the function to optimize
    def func(x):
        return np.sum(x**2)  # a simple, multimodal function

    # Set up the Bayesian Optimization
    bo = Optimize_with_BO(budget=10000, dim=10, initial_points=10, random_seed=42, acquisition_func=ExpectedImprovement)

    # Define the acquisition function using custom_acquisition_function
    def custom_acq_func(x, y, acquisition_func, model):
        f_batch = custom_acquisition_function(x, y, acquisition_func, model)
        return f_batch

    # Define the acquisition function using custom_acq_func and SAWEI
    def custom_saweic_acq_func(x, y, acquisition_func, model):
        f_batch = custom_acquisition_function(x, y, acquisition_func, model)
        sawei_func = acquisition_func.f(x)
        sawei_func = torch.exp(saweic_func) - 1
        return f_batch + sawei_func

    # Perform the optimization
    best_f, best_x = bo(func)

    print(f"Best function value: {best_f}")
    print(f"Best x: {best_x}")

    # Define the custom acquisition function using custom_acq_func and SAWEI
    custom_acq_func = custom_saweic_acq_func
    # Perform the optimization
    best_f, best_x = bo(custom_acq_func, func)

    print(f"Best function value: {best_f}")
    print(f"Best x: {best_x}")