import torch
import torch.nn.functional as F
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from botorch.acquisition.acquisition import AcqFunction
from botorch.acquisition.utils import AcquisitionTask
from botorch.models import GaussianProcessModel
from botorch.optim import optimization_utils
from gpytorch.models import ExactGPR
from gpytorch.likelihoods import GaussianLikelihood
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import scipy.stats as qmc
import torch.nn as nn
import torch.optim as optim

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.f_opt = np.inf
        self.x_opt = None
        self.model = None
        self.acq_func = None
        self.task = None

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

        X = torch.tensor(self.doe_points).float()
        Y = torch.tensor(func(self.doe_points)).float()
        self.model.fit(X, Y)

    def _acq_func_init(self):
        self.acq_func = AcqFunction(self.task)

    def _acq_init(self):
        if self.model is not None:
            self.acq_func = ExpectedImprovement(model=self.model)
            self.acq_func = UpperConfidenceBound(model=self.model)
        elif self.model is None:
            self.acq_func = ProbabilityOfImprovement(model=self.model)

    def _get_new_points(self):
        if self.acq_func is not None:
            new_points = self.acq_func(self.task)
            return new_points, self.task
        else:
            return torch.randn((1, self.dim)), self.task

    def _update_acq_func(self, x, f):
        if self.acq_func is not None:
            x = torch.tensor(x).float()
            f = torch.tensor(f).float()
            self.acq_func(x, f)
            self.f_opt = f.min()
            self.x_opt = x.argmin()

# Refine the strategy by using a more efficient acquisition function
# SAWEI (SAWEI: Smoothly Adapting Weighted Expected Improvement)
class SAWEI:
    def __init__(self, model):
        self.model = model
        self.acq_func = ExpectedImprovement(model=model)
        self.acq_func = UpperConfidenceBound(model=model)

    def __call__(self, task):
        # compute the expected improvement using qExpectedImprovement
        # compute the upper confidence bound using qUpperConfidenceBound
        # and then combine them using the weight of qExpectedImprovement and qUpperConfidenceBound
        qEI = self.acq_func(task)
        qUCB = qUpperConfidenceBound(task, self.model)
        return qEI + 0.5 * qUCB

class Optimize_with_BO_SAWEI(Optimize_with_BO):
    def __init__(self, budget, dim, initial_points, random_seed):
        super().__init__(budget, dim, initial_points, random_seed)
        self.sawe_i = SAWEI(self.model)
        self.sawe_u = UpperConfidenceBound(model=self.model)

    def _get_new_points(self):
        new_points = self.sawe_i(self.task)
        return new_points, self.task

    def _update_acq_func(self, x, f):
        if self.sawe_i is not None:
            x = torch.tensor(x).float()
            f = torch.tensor(f).float()
            self.sawe_i(x, f)
            self.sawe_u(x, f)
            self.f_opt = f.min()
            self.x_opt = x.argmin()