VANILLA_ROLE_PROMPT = """\
You are an AI that generates Python optimization code by using Bayesian Optimization.
"""
VANILLA_TASK_PROMPT = """\
Create a Python class named Optimize_with_BO, with __init__(self, budget, dim, initial_points, random_seed) that initializes the budget and dimension, the initial DoE points and the random seed (for ulterior Sobol samplers from torch or for LHS from scipy),
and a __call__(self, func) method that tries to find x in [-5, 5]^dim that minimizes func(x) (this can be set randomly, but imagine it's a highly multimodal function like the Gallagher 101 peaks).
The function should return the tuple best_f, best_x (value best found and the location).
Implement an efficient algorithm for this Bayesian Optimization task. The code you implement must
only use BoTorch and GPyTorch libraries for the optimization; 
You may also use scipy to generate random samples for the initial DoE and then transform the variables fromlists or numpy arrays into torch tensors. 
Do not use any other optimization libraries. You may append additional helper functions or classes if needed, but the main class must be named Optimize_with_BO and must have the specified methods.
You can also use other BoTorch acquisition functions like qNoisyExpectedImprovement, qLogNoisyExpectedImprovement, qExpectedImprovement, qLogExpectedImprovement, qUpperConfidenceBound, and you can use the MaternKernel and ScaleKernel from GPyTorch.
You can also design your own acquisition function if you want, but it must be based on the BoTorch framework. Check for example the SAWEI from Benjamins et al. (2023) paper, which is a combination of qLogExpectedImprovement and qUpperConfidenceBound.
Or for instance take a look at the paper from HEBO and implement a GP-Hedge balancing the acquisition functions.

An example of such code (a simple random search), is as follows:
```
import numpy as np

class RandomSearch:
    def __init__(self, budget: int, dim: int, initial_points: int, random_seed: int):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed(self.random_seed)
        for i in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)

            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

        return self.f_opt, self.x_opt
```

An example of valid imports:
```
from gpytorch.likelihoods import GaussianLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
```

Provide the Python code and a one-line description with the main idea (without enters). Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```

IMPORTANT TENSOR SHAPE REQUIREMENTS FOR BOTORCH:
- Training inputs `train_X` must be 2D tensors of shape `(n_samples, dim)`.
- Training targets `train_Y` must be strictly 2D tensors of shape `(n_samples, 1)`. Always use `train_Y = train_Y.unsqueeze(-1)` if it is 1D before passing it to BoTorch models.
- Avoid deprecated NumPy/PyTorch conversions; ensure tensors are properly cloned using `sourceTensor.clone().detach()`.
"""