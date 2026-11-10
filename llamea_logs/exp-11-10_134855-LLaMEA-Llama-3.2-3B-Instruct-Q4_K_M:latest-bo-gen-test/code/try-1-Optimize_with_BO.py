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
from scipy.stats import norm

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
        self.init_doelike_points()

    def init_doelike_points(self):
        np.random.seed(self.random_seed)
        self.sampler = SobolSet(d=self.dim, q=self.initial_points)
        self.mcs = set_default_mcs()
        self.doelike_points = self.sampler.evaluate(f=lambda x: torch.tensor(x).float())
        self.doelike_points = torch.from_numpy(self.doelike_points).float()
        self.x_min = torch.tensor([-5] * self.dim).float()
        self.x_max = torch.tensor([5] * self.dim).float()

    def fit(self, func):
        # Create a Gaussian process model with Matern kernel
        likelihood = MultitaskMultivariateNormalLikelihood(num_tasks=1)
        self.gp_model = ExactGP(self.doelike_points, likelihood, self.mcs)
        self.gp_model.train_likewise(self.doelike_points)

        # Choose an acquisition function (in this case, qExpectedImprovement)
        self.acquisition = qExpectedImprovement(self.gp_model, self.x_min, self.x_max)

        # Perform Bayesian optimization
        for i in range(self.budget):
            # Compute the posterior expectations for the current DoE points
            posterior_expectations = self.gp_model.posterior(self.doelike_points)
            # Compute the expected improvement for each DoE point
            expected_improvements = self.acquisition(expected_posterior=posterior_expectations)
            # Get the index of the DoE point with the highest expected improvement
            best_idx = torch.argmax(expected_improvements)
            # Get the location of the best DoE point
            best_x = self.doelike_points[best_idx].item()
            # Evaluate the function at the best DoE point
            best_f = func(best_x)
            # Update the best point and its corresponding function value
            if best_f < torch.tensor(self.f_opt).float():
                self.f_opt = best_f
                self.x_opt = best_x
            else:
                print(f"Failed to find better point: best_f={best_f:.4f}, best_x={best_x:.4f}")
            print(f"Iteration {i+1}, best_x={best_x:.4f}, best_f={best_f:.4f}")

        # Return the final best point and its corresponding function value
        return self.f_opt, self.x_opt

    def postfit_acqfunc(self, func):
        # Plot the final acquisition function
        import matplotlib.pyplot as plt
        x = torch.linspace(self.x_min, self.x_max, 1000).float()
        f_x = torch.tensor([func(x_i).item() for x_i in x]).float()
        acquisition_func = self.acquisition(f_x)
        plt.plot(x, acquisition_func.detach().numpy(), label='Acquisition function')
        plt.plot(x, f_x, label='Function')
        plt.legend()
        plt.show()

        # Compute the final expected improvement
        posterior_expectations = self.gp_model.posterior(self.doelike_points)
        expected_improvements = self.acquisition(expected_posterior=posterior_expectations)
        final_expected_improvement = torch.max(expected_improvements)
        print(f"Final expected improvement: {final_expected_improvement:.4f}")

# Initialize the Optimize_with_BO class
optimization = Optimize_with_BO(budget=100, dim=10, initial_points=50, random_seed=42)

# Define the function to be optimized
def func(x):
    # For demonstration purposes, use the Sum of Squares of Linear Functions
    return (torch.sum(x**2, dim=1) - torch.mean(x**2, dim=1))**2

# Perform Bayesian optimization
optimization.fit(func)

# Post-fitting acquisition function
optimization.postfit_acqfunc(func)