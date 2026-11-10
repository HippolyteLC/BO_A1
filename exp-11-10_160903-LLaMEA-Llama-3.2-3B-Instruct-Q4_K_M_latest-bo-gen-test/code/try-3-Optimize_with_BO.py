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
from scipy.stats import qmc

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
        self.sobol_dim = dim

    def __call__(self, func):
        torch.manual_seed(self.random_seed)
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
            self.doe = qmc.LatinHypercube(dim=self.sobol_dim, scale=1)
            self.doe_points = self.doe.random(self.initial_points)
            self.doe_points = torch.tensor(self.doe_points).float()
            self.doe_task = AcquisitionTask(self.doe_points, self.sobol_dim)
            self.task = self.doe_task
        else:
            self.doe_points = torch.tensor(np.random.uniform(low=-5, high=5, size=(self.initial_points, self.sobol_dim))).float()

    def _train_gpr(self):
        if self.model is None:
            self.model = GaussianProcessModel()
            likelihood = GaussianLikelihood()
            self.model.likelihood = likelihood
            self.model.fit(self.doe_points, func(self.doe_points))

    def _acq_func_init(self):
        if self.acq_func is None:
            self.acq_func = ExpectedImprovement(model=self.model)
            self.acq_func = UpperConfidenceBound(model=self.model)

    def _acq_init(self):
        self.acq_func = AcqFunction(self.task)

    def _get_new_points(self):
        new_points = self.acq_func(self.task)
        return new_points, self.task

    def _update_acq_func(self, x, f):
        x = torch.tensor(x).float()
        f = torch.tensor(f).float()
        self.acq_func(x, f)
