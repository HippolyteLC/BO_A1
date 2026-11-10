import torch
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement, qNoisyExpectedImprovement, qLogNoisyExpectedImprovement, qExpectedImprovement, qLogExpectedImprovement, qUpperConfidenceBound
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from torch.distributions import MultivariateNormal
from scipy.stats import qmc
import numpy as np

class Optimize_with_BO:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.f_opt = np.inf
        self.x_opt = None
        self.acq = ExpectedImprovement()
        self.gp_model = ExactGP(dim=self.dim, likelihood=GaussianLikelihood())
        self.sampler = qmc.HastieHaslett(dim=self.dim, seed=self.random_seed)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        for i in range(self.initial_points):
            x = torch.tensor(self.sampler(i).to(0), dtype=torch.float32)
            y = torch.tensor(np.random.uniform(-5, 5), dtype=torch.float32)
            self.gp_model.fit(y.view(1, self.dim), x.view(1, self.dim))

    def __call__(self, func):
        for i in range(self.budget):
            x = torch.tensor(self.sampler.sample(1), dtype=torch.float32).to(0)
            f = func(x)
            with torch.no_grad():
                y_pred, _ = self.gp_model(x)
                acq_func = self.acq(y_pred, x, self.gp_model.likelihood)
                z = torch.tensor(acq_func.scores[0], dtype=torch.float32)
            if z < 0:
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = x
        return self.f_opt, self.x_opt