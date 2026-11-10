import torch
import torch.nn.functional as F
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.acquisition import AcqFunction
from botorch.acquisition.utils import AcquisitionTask
from botorch.models import GaussianProcessModel
from botorch.optim import optimization_utils
from gpytorch.models import ExactGPR
from gpytorch.likelihoods import GaussianLikelihood
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math
import gpytorch
from torch.distributions import StudentT
from scipy.stats import qmc

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed, kernel, acquisition_func):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.f_opt = np.inf
        self.x_opt = None
        self.model = None
        self.acq_func = None
        self.task = None
        self.kernel = kernel
        self.acquisition_func = acquisition_func

    def __call__(self, func):
        np.random.seed(self.random_seed)
        self._initialize_doe()
        self._train_gpr()
        self._acq_func_init()
        self._acq_init()

        for i in range(self.budget):
            x, _ = self._get_new_points()
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            self._update_acq_func(x, f)

        return self.f_opt, self.x_opt

    def _initialize_doe(self):
        if self.initial_points > 0:
            self.doe = qmc.LatinHypercube(dim=self.dim, scale=1)
            self.doe_points = self.doe.random(self.initial_points)
            self.doe_points = torch.tensor(self.doe_points).float()
            self.doe_task = AcquisitionTask(self.doe_points, self.dim)
            self.task = self.doe_task
        else:
            self.doe_points = torch.tensor(np.random.uniform(low=-5, high=5, size=(self.initial_points, self.dim))).float()

    def _train_gpr(self):
        if self.model is None:
            self.model = GaussianProcessModel()
            likelihood = GaussianLikelihood()
            self.model.likelihood = likelihood
            kernel = self.kernel
            self.model.kernel = kernel

        X = torch.tensor(self.doe_points).float()
        Y = torch.tensor(func(self.doe_points)).float()
        self.model.fit(X, Y)

    def _acq_func_init(self):
        if self.acq_func is None:
            if self.acquisition_func == 'ExpectedImprovement':
                self.acq_func = ExpectedImprovement(model=self.model)
            elif self.acquisition_func == 'UpperConfidenceBound':
                self.acq_func = UpperConfidenceBound(model=self.model)
            elif self.acquisition_func == 'SAWEI':
                self.acq_func = (self.model.mean_function(torch.tensor(self.doe_points).float()) - self.model.mean_function(torch.tensor(self.doe_points).float())).max()

    def _acq_init(self):
        self.acq_func = AcqFunction(self.task)

    def _get_new_points(self):
        new_points = self.acq_func(self.task)
        return new_points, self.task

    def _update_acq_func(self, x, f):
        x = torch.tensor(x).float()
        f = torch.tensor(f).float()
        self.acq_func(x, f)

# Define kernel and acquisition function
class MaternKernel(gpytorch.kernels.Kernel):
    def __init__(self, ard_num_dims=1):
        super(MaternKernel, self).__init__(ard_num_dims=ard_num_dims)

    def _kernel(self, x, y):
        dist = x.size(0)
        if dist == 0:
            return torch.zeros_like(y)
        if dist == 1:
            return self._matern32(x)
        if dist == 2:
            return self._matern32(x).view(1, -1) * self._matern32(y).view(-1, 1)
        if dist == 3:
            return self._matern32(x).view(1, -1) * self._matern32(y).view(-1, 1) + self._matern32(y).view(-1, 1)
        dists = torch.sqrt(torch.sum(x**2, dim=-1) + torch.sum(y**2, dim=-1)) - 2 * torch.matmul(x, y.transpose(-1, -2))
        dists = torch.clamp(dists, min=0)
        dists = 1 / dists
        dists = torch.pow(dists, 1/3)
        return self._matern32(x) * self._matern32(y) * dists

class SAWEI(AcquisitionFunction):
    def __init__(self, model, acq_func=None):
        super(SAWEI, self).__init__(model, acq_func)

    def acq_func(self, task):
        x = task.x
        mean = self.model.mean_function(torch.tensor(x).float())
        variance = self.model.variance_function(torch.tensor(x).float())
        upper_bound = mean + torch.sqrt(variance) * torch.ones_like(mean)
        lower_bound = mean - torch.sqrt(variance) * torch.ones_like(mean)
        return torch.cat((upper_bound - mean, lower_bound - mean)).min()

def custom_acq_func(x, f):
    return (x - f).max()

# Example usage
def func(x):
    return -x[0]**2 - x[1]**2

opt = Optimize_with_BO(budget=10, dim=2, initial_points=10, random_seed=42, kernel=MaternKernel(), acquisition_func='SAWEI')
print(opt(func))