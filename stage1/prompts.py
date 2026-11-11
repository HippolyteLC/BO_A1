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
ENGINEERED_TASK_PROMPT = """Create a Python class named Optimize_with_BO, with __init__(self, budget, dim, initial_points, random_seed) that stores the budget, dimension, number of initial points and random seed.
It must implement __call__(self, func), which minimizes func(x) over the box given by func.bounds.lb and func.bounds.ub (here [-5, 5]^dim) and returns the tuple (best_f, best_x): the lowest objective value found and its location as a numpy array.

### Problem facts
- func is a MINIMIZATION problem. It takes a numpy array of shape (dim,) and returns a float.
- Objective values are non-negative, span several orders of magnitude (from about 0 to over 10000) and are heavy-tailed. The function is multimodal.
- You may call func at most `budget` times in total, including the initial design. Never exceed it. Results are scored by the area over the convergence curve, so finding good points early is rewarded.

### Implementation requirements
1. Use ONLY BoTorch, GPyTorch, PyTorch, NumPy and SciPy. No other optimization libraries.
2. Generate the initial design with torch.quasirandom.SobolEngine (use the argument `scramble=True`, NOT `randomized`) or SciPy LHS.
3. BoTorch MAXIMIZES. To minimize func you must fit the surrogate on a negated target and use best_f = max of that negated target.
4. Work in the unit cube [0, 1]^dim internally and rescale to [lb, ub] only when calling func.
5. Transform the targets before fitting: log10(1 + y), standardize, then negate.
6. Use SingleTaskGP, ExactMarginalLogLikelihood, fit_gpytorch_mll and optimize_acqf. You may choose a different BoTorch acquisition function, kernel, or acquisition strategy to improve performance.

### Working reference skeleton (follow this pattern)
```python
import numpy as np
import torch
from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int, random_seed: int):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed

    def _evaluate(self, func, x_unit, lb, ub):
        x = lb + (ub - lb) * x_unit
        return float(func(x.detach().cpu().numpy()))

    def __call__(self, func):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        lb = torch.as_tensor(np.asarray(func.bounds.lb, dtype=float).reshape(-1), dtype=torch.double)
        ub = torch.as_tensor(np.asarray(func.bounds.ub, dtype=float).reshape(-1), dtype=torch.double)
        unit_bounds = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)]).to(torch.double)

        n_init = min(self.initial_points, self.budget)
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.random_seed)
        X = sobol.draw(n_init).to(torch.double)
        Y = torch.tensor([self._evaluate(func, x, lb, ub) for x in X], dtype=torch.double)

        while Y.shape[0] < self.budget:
            z = torch.log10(1.0 + Y.clamp_min(0.0))
            z = (z - z.mean()) / (z.std() + 1e-9)
            train_Y = (-z).unsqueeze(-1)  # shape (n, 1), larger is better

            model = SingleTaskGP(X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            acq = qLogExpectedImprovement(model=model, best_f=train_Y.max())
            cand, _ = optimize_acqf(acq, bounds=unit_bounds, q=1, num_restarts=5, raw_samples=64)
            x_new = cand.detach().reshape(1, -1)

            y_new = self._evaluate(func, x_new.squeeze(0), lb, ub)
            X = torch.cat([X, x_new], dim=0)
            Y = torch.cat([Y, torch.tensor([y_new], dtype=torch.double)])

        i = int(torch.argmin(Y))
        best_x = (lb + (ub - lb) * X[i]).numpy()
        return float(Y[i]), best_x
```

### Critical rules
- Never mix NumPy arrays and torch tensors in arithmetic. Convert with torch.as_tensor(..., dtype=torch.double).
- train_X has shape (n, dim) and train_Y has shape (n, 1).
- Use exactly these imports for fitting: `from botorch.fit import fit_gpytorch_mll`.

Provide the Python code and a one-line description with the main idea (without enters). Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```
"""

OPEN_TASK_PROMPT = """Create a Python class named Optimize_with_BO that minimizes a black-box function using Bayesian Optimization.
Your goal is to design an algorithm that finds good solutions as early as possible, because results are scored by the area over the convergence curve.

### Problem facts
- func is a MINIMIZATION problem. It takes a numpy array of shape (dim,) and returns a float.
- The search box is func.bounds.lb to func.bounds.ub (here [-5, 5]^dim, dim = 6).
- Objective values are non-negative, heavy-tailed (from about 0 to over 10000) and the function is multimodal.
- You may call func at most `budget` times in total, including the initial design (budget = 60, initial_points = 10 in practice). Never exceed it.

### Class structure (keep these names and signatures, fill in the bodies)
```python
class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int, random_seed: int):
        \"\"\"Store the arguments. Add any hyperparameters of your own design here.\"\"\"

    def _initial_design(self, n_points: int) -> torch.Tensor:
        \"\"\"Return n_points starting locations as a double tensor of shape (n_points, dim)
        in whatever coordinate system you choose to work in.\"\"\"

    def _fit_model(self, X: torch.Tensor, y: torch.Tensor):
        \"\"\"X has shape (n, dim). y holds raw objective values (minimization) of shape (n,).
        Decide how to transform y, choose the surrogate and kernel, fit it, and return the fitted model.\"\"\"

    def _select_next_point(self, model, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        \"\"\"Return the next location to evaluate as a tensor of shape (1, dim).
        Decide the acquisition function or strategy and how to optimize it.\"\"\"

    def __call__(self, func):
        \"\"\"Run the optimization loop within the budget.
        Return (best_f, best_x): best_f is a float, best_x a numpy array of shape (dim,).\"\"\"
```
You may add helper methods or classes. The main class name, `__init__` signature and `__call__` return format must stay as specified.

### Be creative within these limits
Choose a design that goes beyond plain Expected Improvement with a default GP. For example, consider Thompson sampling, a trust region around the incumbent, a portfolio of acquisition functions, a different kernel, an input or output transformation, or an adaptive exploration schedule. Pick one coherent idea and implement it fully.

### Hard rules (these avoid common crashes)
- Use ONLY BoTorch, GPyTorch, PyTorch, NumPy and SciPy.
- BoTorch MAXIMIZES. If you fit the model on minimization values, negate the target (or otherwise make it a maximization target) before building the acquisition function, and compute best_f accordingly.
- train_X must have shape (n, dim) and train_Y must have shape (n, 1). Use .unsqueeze(-1) on 1D targets.
- Use torch.double everywhere. Never mix NumPy arrays and tensors in arithmetic; convert with torch.as_tensor(..., dtype=torch.double). Use .detach() before calling .numpy().
- Correct imports:
  from botorch.models import SingleTaskGP
  from botorch.fit import fit_gpytorch_mll   # NOT fit_gpytorch_model
  from botorch.optim import optimize_acqf
  from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
  from gpytorch.mlls import ExactMarginalLogLikelihood
  from gpytorch.kernels import MaternKernel, ScaleKernel
- SobolEngine takes `scramble=True`, NOT `randomized`: torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed).
- Raw objective values span several orders of magnitude, so consider a transform such as log10(1 + y) plus standardization before fitting. Working in the unit cube and rescaling only when calling func also helps numerically.
- Set torch.manual_seed and np.random.seed from random_seed.
- Do not wrap the whole loop in a try/except that falls back to random search.

Provide the Python code and a one-line description with the main idea (without enters). Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```
"""