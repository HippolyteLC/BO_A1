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

# Description: Role and Task Prompts for LLaMEA BO Algorithm Generation

# ==============================================================================
# ROLE PROMPT
# ==============================================================================
ENGINEERED_ROLE_PROMPT = """You are an AI that generates Python optimization code by using Bayesian Optimization.
"""

# ==============================================================================
# TASK PROMPT
# ==============================================================================
ENGINEERED_TASK_PROMPT = """Create a Python class named Optimize_with_BO, with __init__(self, budget, dim, initial_points, random_seed) that initializes the budget, dimension, initial points count, and random seed.
It must also implement a __call__(self, func) method that tries to find x in [-5, 5]^dim that minimizes func(x).
The __call__ function must return the tuple best_f, best_x (the best objective value found and its location tensor/array).

### Implementation Requirements:
1. **Library Constraints:** Use ONLY BoTorch, GPyTorch, PyTorch, and SciPy. Do not use any other optimization libraries.
2. **Initialization:** Use `torch.quasirandom.SobolEngine` (or SciPy LHS) to generate initial design points within `func.bounds.lb` and `func.bounds.ub`. Ensure bounds are converted to torch tensors immediately.
3. **Surrogate & Acquisition:** Use `SingleTaskGP`, `ExactMarginalLogLikelihood`, `fit_gpytorch_model`, and a standard BoTorch acquisition function (e.g., `qExpectedImprovement` or `qNoisyExpectedImprovement`).
4. **Optimization Loop:** Iteratively fit the GP model, optimize the acquisition function using `optimize_acqf` over the bounds `[-5, 5]`, evaluate the new point, and update the training tensors `train_X` and `train_Y` until the `budget` is exhausted.

### Working Reference Skeleton (Follow this pattern strictly):
```python
import torch
import numpy as np
from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf

class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int, random_seed: int):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed

    def __call__(self, func):
        torch.manual_seed(self.random_seed)
        
        # 1. Get bounds as torch tensors
        lb = torch.tensor(func.bounds.lb, dtype=torch.double)
        ub = torch.tensor(func.bounds.ub, dtype=torch.double)

        # 2. Initial Design of Experiments (DoE) using Sobol
        sobol = SobolEngine(dimension=self.dim, randomized=True, seed=self.random_seed)
        train_X = sobol.draw(self.initial_points).to(torch.double)
        train_X = lb + (ub - lb) * train_X  # Safe tensor scaling (no numpy mixing)

        # Evaluate initial points
        train_Y_list = []
        for x in train_X:
            val = func(x.numpy())
            train_Y_list.append(val)
        
        train_Y = torch.tensor(train_Y_list, dtype=torch.double).unsqueeze(-1) # Must be [N, 1]

        best_idx = torch.argmin(train_Y)
        best_f = train_Y[best_idx].item()
        best_x = train_X[best_idx].numpy()

        evals_done = self.initial_points

        # 3. BO Loop
        while evals_done < self.budget:
            # Fit GP model
            model = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            # Define Acquisition Function
            acq_func = qExpectedImprovement(model=model, best_f=train_Y.min())

            # Optimize Acquisition Function
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.stack([lb, ub]),
                q=1,
                num_restarts=5,
                raw_samples=32,
            )

            new_x = candidates.detach()
            new_y_val = func(new_x.squeeze(0).numpy())
            new_y = torch.tensor([[new_y_val]], dtype=torch.double)

            # Update dataset
            train_X = torch.cat([train_X, new_x], dim=0)
            train_Y = torch.cat([train_Y, new_y], dim=0)

            evals_done += 1

            if new_y_val < best_f:
                best_f = new_y_val
                best_x = new_x.squeeze(0).numpy()

        return best_f, best_x
```

### CRITICAL RULES TO AVOID ERRORS:
- **Never mix NumPy and Tensors:** Do not multiply a numpy array by a tensor directly. Convert everything to `torch.tensor` using `.double()` or `.to(torch.double)` right away.
- **Tensor Shapes:** `train_Y` **must** always have shape `(n_samples, 1)`. Always use `.unsqueeze(-1)` on target vectors.
- **Correct Imports:** 
  - `from botorch.fit import fit_gpytorch_model` (Never import this directly from `botorch`).
  - `from botorch.models import SingleTaskGP`
  - `from gpytorch.mlls import ExactMarginalLogLikelihood`
  - `from botorch.optim import optimize_acqf`

Provide the Python code and a one-line description with the main idea (without enters). Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```
"""