import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Distribution
from gpytorch.models import ExactGP
from gpytorch.lazy import LazyGP
from gpytorch.models import GaussianProcess
from botorch.models import VaryingDGP
from botorch.models import VaryingDGP
from botorch.acquisition import SAWEI
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition import qUpperConfidenceBound
from torch.distributions import Normal

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed, acq_method=SAWEI(qExpectedImprovement, qUpperConfidenceBound)):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.acq_method = acq_method
        self.gp = ExactGP(dim=self.dim, likelihood=Normal, prior=Normal)
        self.acq = AcquisitionFunction(self.acq_method)
        self.x_train = None
        self.y_train = None

    def __call__(self, func):
        np.random.seed(self.random_seed)
        for i in range(self.initial_points):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)

            f = func(x)
            if f < self.y_train[-1] if self.y_train else f:
                self.y_train = np.array([f])
                self.x_train = np.array([x])
            else:
                self.y_train = np.append(self.y_train, f)
                self.x_train = np.vstack((self.x_train, x))

        self.gp.fit(self.x_train, self.y_train)
        while self.budget > 0:
            best_x = torch.tensor(self.x_train[np.argmin(self.y_train)], requires_grad=False)
            best_f = self.y_train[np.argmin(self.y_train)]
            acquisition_value = self.acq(self.gp, best_x)
            if acquisition_value < self.acq_method(qExpectedImprovement, qUpperConfidenceBound)(self.gp, best_x):
                next_point = self.acq_method(qExpectedImprovement, qUpperConfidenceBound).get_next_point(best_x)
                self.x_train = np.vstack((self.x_train, next_point))
                self.y_train = np.append(self.y_train, func(next_point))
                self.gp.fit(self.x_train, self.y_train)
                self.budget -= 1
            else:
                break
        return torch.tensor(best_f, requires_grad=False), best_x