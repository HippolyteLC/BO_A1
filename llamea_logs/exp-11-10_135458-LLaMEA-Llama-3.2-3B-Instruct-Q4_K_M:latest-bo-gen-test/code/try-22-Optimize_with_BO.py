import torch
import torch.nn as nn
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.acquisition import qExpectedImprovement
import numpy as np
from scipy.special import root
from scipy.stats import norm

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.x_train = None
        self.y_train = None
        self.gp_model = None
        self.acq_fn = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        torch.manual_seed(self.random_seed)

    def initialize_doe(self, func, n_init=initial_points):
        np.random.seed(self.random_seed)
        x_init = np.random.uniform(func.bounds.lb, func.bounds.ub, (n_init, self.dim))
        y_init = func(x_init)
        self.x_train = torch.from_numpy(x_init).to(self.device)
        self.y_train = torch.from_numpy(y_init).to(self.device)

    def fit_gp(self):
        self.gp_model = ExactGP(self.device, GaussianLikelihood(), self.x_train, self.y_train)
        self.gp_model.train()
        self.acq_fn = qExpectedImprovement(self.gp_model, self.device)

    def optimize(self, func):
        self.initialize_doe(func)
        self.fit_gp()
        best_f, best_x = self.x_train.max(dim=0).values().item(), self.x_train.max(dim=0).values().item()
        for _ in range(self.budget):
            x = self.acq_fn(self.x_train)
            x = torch.tensor(x).to(self.device)
            f = func(x)
            x = x.detach().cpu().numpy()
            if f < best_f:
                best_f, best_x = f, x
            self.x_train = torch.cat((self.x_train, x))
            self.y_train = torch.cat((self.y_train, f))
            self.fit_gp()
        return best_f, best_x