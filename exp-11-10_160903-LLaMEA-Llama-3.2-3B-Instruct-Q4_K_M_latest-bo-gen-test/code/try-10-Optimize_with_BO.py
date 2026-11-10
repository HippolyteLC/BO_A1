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
        Y = torch.tensor(func(X)).float()
        self.model.fit(X, Y)

    def _acq_func_init(self):
        if self.acq_func is None:
            # Fix the import error
            from botorch.acquisition.acquisition import AcqFunction
            self.acq_func = AcqFunction(task=self.task)

    def _acq_init(self):
        self.acq_func = ExpectedImprovement(model=self.model)

    def _get_new_points(self):
        new_points = self.acq_func(self.task)
        return new_points, self.task

    def _update_acq_func(self, x, f):
        x = torch.tensor(x).float()
        f = torch.tensor(f).float()
        # Use the updated acquisition function to update the model and task
        self.model.update_data(torch.tensor([x]).float(), torch.tensor([f]).float())
        self.task.update_data(torch.tensor([x]).float(), torch.tensor([f]).float())

    # Implementing the self-heating strategy
    def self_heating(self, max_iter=10):
        for _ in range(max_iter):
            # Check if the current point is the best
            if self.f_opt == func(self.x_opt):
                # If it is, generate new points and update the acquisition function
                new_points, self.task = self._get_new_points()
                self._update_acq_func(new_points, func(new_points))
            else:
                # If it's not, stop the self-heating process
                break

    # Implementing the SAWEI strategy
    def sawei(self, max_iter=10, alpha=0.5):
        for _ in range(max_iter):
            # Generate new points using the SAWEI method
            new_points, self.task = self._get_new_points()
            # Update the acquisition function and model
            self._update_acq_func(new_points, func(new_points))
            # Update the task
            self.task.update_data(torch.tensor([new_points]).float(), torch.tensor([func(new_points)].float()))
            # Apply the weighting scheme
            weights = torch.ones_like(new_points)
            weights[torch.argmax(func(new_points))] = alpha
            new_points = torch.multiply(new_points, weights)
            # Update the acquisition function and model
            self._update_acq_func(new_points, func(new_points))

# Usage:
def func(x):
    return x**2 + 0.1 * np.sin(x**2)

opt = Optimize_with_BO(budget=10, dim=1, initial_points=5, random_seed=42)
best_f, best_x = opt(func)
print(f"Best function value: {best_f}")
print(f"Best x: {best_x}")