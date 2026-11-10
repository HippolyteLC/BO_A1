import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from gpytorch.models import ExactGP
from gpytorch Likelihoods import MultitaskMultivariateNormalLikelihood
from gpytorch.models import set_default_kernel
from gpytorch.kernels import ScaleKernel, MaternKernel
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition import qUpperConfidenceBound
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import minimize_posterior
from botorch.sampling import SobolSet
from botorch.utils import set_default_mcs

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.gp_model = None
        self.acquisition = None
        self.sampler = None
        self.mcs = None
        self.x_min = torch.tensor([-5] * self.dim).float()
        self.x_max = torch.tensor([5] * self.dim).float()
        self.f_opt = torch.inf
        self.x_opt = None

    def init_doelike_points(self):
        np.random.seed(self.random_seed)
        self.sampler = SobolSet(d=self.dim, q=self.initial_points)
        self.mcs = set_default_mcs()
        self.doelike_points = self.sampler.evaluate(f=lambda x: torch.tensor(x).float())
        self.doelike_points = torch.from_numpy(self.doelike_points).float()

    def fit(self, func):
        likelihood = MultitaskMultivariateNormalLikelihood(num_tasks=1)
        self.gp_model = ExactGP(self.doelike_points, likelihood, self.mcs)
        self.gp_model.train_likewise(self.doelike_points)

        self.acquisition = qExpectedImprovement(self.gp_model, self.x_min, self.x_max)

        for i in range(self.budget):
            posterior_expectations = self.gp_model.posterior(self.doelike_points)
            expected_improvements = self.acquisition(expected_posterior=posterior_expectations)
            best_idx = torch.argmax(expected_improvements)
            best_x = self.doelike_points[best_idx].item()
            best_f = func(best_x)

            if best_f < self.f_opt:
                self.f_opt = best_f
                self.x_opt = best_x
            else:
                print(f"Failed to find better point: best_f={best_f:.4f}, best_x={best_x:.4f}")
            print(f"Iteration {i+1}, best_x={best_x:.4f}, best_f={best_f:.4f}")

        return self.f_opt, self.x_opt

    def optimize(self, func, acquisition_function='qExpectedImprovement', kernel='MaternKernel'):
        self.init_doelike_points()
        self.gp_model = None
        self.acquisition = None

        likelihood = MultitaskMultivariateNormalLikelihood(num_tasks=1)
        if kernel == 'MaternKernel':
            kernel = MaternKernel()
        else:
            kernel = ScaleKernel()

        self.gp_model = ExactGP(self.doelike_points, likelihood, kernel)
        self.gp_model.train_likewise(self.doelike_points)

        self.acquisition = qExpectedImprovement(self.gp_model, self.x_min, self.x_max, acquisition_function=acquisition_function)
        self.acquisition = qLogExpectedImprovement(self.gp_model, self.x_min, self.x_max, acquisition_function=acquisition_function)

        for i in range(self.budget):
            posterior_expectations = self.gp_model.posterior(self.doelike_points)
            expected_improvements = self.acquisition(expected_posterior=posterior_expectations)
            best_idx = torch.argmax(expected_improvements)
            best_x = self.doelike_points[best_idx].item()
            best_f = func(best_x)

            if best_f < self.f_opt:
                self.f_opt = best_f
                self.x_opt = best_x
            else:
                print(f"Failed to find better point: best_f={best_f:.4f}, best_x={best_x:.4f}")
            print(f"Iteration {i+1}, best_x={best_x:.4f}, best_f={best_f:.4f}")

        return self.f_opt, self.x_opt