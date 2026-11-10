import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Distribution
from gpytorch.models import ExactGP
from gpytorch.lazy import LazyGP
from gpytorch.models import GaussianProcess
from gpytorch.lazy import LazyGP
from gpytorch.models import VaryingDP
from gpytorch.priors import ConstantPrior, IdentityPrior, HalfNormalPrior
from gpytorch.lazy import LazyGP
from gpytorch.models import ExactGP
from gpytorch.lazy import LazyGP
from botorch.models import VaryingDGP
from botorch import AcquisitionFunction, AcquisitionMethod
from botorch.models import VaryingDGP

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed, acq_method=AcquisitionMethod(qExpectedImprovement)):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.acq_method = acq_method
        self.gp = ExactGP(dim=self.dim, likelihood=torch.distributions.Normal, prior=torch.distributions.Normal)
        self.acq = AcquisitionFunction(self.acq_method)
        self.x_train = None
        self.y_train = None

    def __call__(self, func):
        self.x_train = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.initial_points, self.dim))
        self.y_train = func(self.x_train)
        self.gp.fit(self.x_train, self.y_train)
        while self.budget > 0:
            best_x = torch.tensor(self.x_train[np.argmin(self.y_train)], requires_grad=False)
            best_f = self.y_train[np.argmin(self.y_train)]
            self.acq(self.gp, best_x, best_f)
            next_point = self.acq.get_next_point(best_x)
            self.x_train = np.vstack((self.x_train, next_point))
            self.y_train = np.append(self.y_train, func(next_point))
            self.gp.fit(self.x_train, self.y_train)
            self.budget -= 1
        return torch.tensor(best_f, requires_grad=False), best_x