VANILLA_ROLE_PROMPT = """\
You are an AI that generates Python optimization code by using Bayesian Optimization.
"""
VANILLA_TASK_PROMPT = """\
Create a Python class named Optimize_with_BO, with __init__(self, budget, dim, initial_points, random_seed) that initializes the budget and dimension, the initial DoE points and the random seed (for ulterior Sobol samplers from torch or for LHS from scipy),
and a __call__(self, func) method that tries to find x in [-5, 5]^dim that minimizes func(x) (this can be set randomly, but imagine it's a highly multimodal function like the Gallagher 101 peaks).
The function should return the tuple best_f, best_x (value best found and the location).
Implement an efficient algorithm for this Bayesian Optimization task. The code you implement must
only use BoTorch and GPyTorch libraries for the optimization; 
You may also use scipy to generate random samples for the initial DoE and then transform the variables from lists or numpy arrays into torch tensors. 
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

Provide the Python code and a one-line description with the main idea (without enters). Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```
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

BO_TEMPLATE_TASK_PROMPT = """Design a novel and efficient Bayesian Optimization algorithm that minimizes a black-box function.
Results are scored by the area over the convergence curve, so finding good points early is rewarded.

### Problem facts
- func is a MINIMIZATION problem. It takes a numpy array of shape (dim,) and returns a float.
- The search box is [-5, 5]^dim (dim = 6). It equals func.bounds.lb and func.bounds.ub.
- Objective values are non-negative, heavy-tailed (from about 0 to over 10000) and the landscape is multimodal.
- You may call func at most `budget` times in total, including the initial design (budget = 60, initial_points = 10 in practice). Never exceed it.

### Code template (keep the class name, method names, signatures and return formats; fill in the bodies)
```python
import numpy as np
import torch
from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel

class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int, random_seed: int):
        self.budget = budget
        self.dim = dim
        self.random_seed = random_seed
        # bounds has shape (2, dim): bounds[0] lower, bounds[1] upper
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        # X has shape (n_points, dim), y has shape (n_points, 1); keep both as numpy arrays
        self.X = None
        self.y = None
        self.n_evals = 0
        self.n_init = min(initial_points, budget)
        # add your own hyperparameters here, always with default values

    def _sample_points(self, n_points):
        # Initial design or candidate sampling (Sobol with scramble=True, LHS, ...).
        # Return a numpy array of shape (n_points, dim) inside self.bounds.

    def _fit_model(self, X, y):
        # Decide how to transform y (raw values are heavy-tailed), choose the surrogate and kernel,
        # fit it and return the fitted model. Convert X and y to torch.double tensors inside this method.
        # The surrogate is trained on a MAXIMIZATION target if you use BoTorch acquisition functions.

    def _acquisition_function(self, X):
        # Evaluate your acquisition function for the points in X.
        # Return an array of shape (n_points, 1).

    def _select_next_points(self, batch_size):
        # Choose the next points to evaluate by optimizing or sampling the acquisition function.
        # You may use optimize_acqf, random candidates, a trust region around the incumbent, or a hybrid.
        # Return a numpy array of shape (batch_size, dim) inside self.bounds.

    def _evaluate_points(self, func, X):
        # Evaluate func on each row of X. Never evaluate more than self.budget - self.n_evals points
        # (truncate X if needed). Update self.n_evals.
        # Return an array of shape (n_points, 1).

    def _update_eval_points(self, new_X, new_y):
        # Append new_X and new_y to self.X and self.y (create them if None).

    def __call__(self, func):
        # Main minimization loop:
        #   X_init = self._sample_points(self.n_init)
        #   y_init = self._evaluate_points(func, X_init)
        #   self._update_eval_points(X_init, y_init)
        #   while self.n_evals < self.budget:
        #       model = self._fit_model(self.X, self.y)
        #       X_next = self._select_next_points(1)
        #       y_next = self._evaluate_points(func, X_next)
        #       self._update_eval_points(X_next, y_next)
        # !!! Do not call func anywhere except inside _evaluate_points. !!!
        # Return (best_f, best_x): best_f is a float, best_x a numpy array of shape (dim,).
```

### Be creative within these limits
Choose ONE coherent idea and implement it fully, for example a trust region around the incumbent, Thompson sampling, a portfolio of acquisition functions, a different kernel, an output transformation, or an adaptive exploration schedule.
Do not just copy plain Expected Improvement with a default GP.

### Hard rules (these avoid common crashes)
- Use ONLY BoTorch, GPyTorch, PyTorch, NumPy and SciPy.
- BoTorch MAXIMIZES. If you fit the model on minimization values, negate the target before building the acquisition function and compute best_f accordingly.
- train_X must have shape (n, dim) and train_Y must have shape (n, 1). Use .unsqueeze(-1) on 1D targets.
- Use torch.double everywhere. Never mix NumPy arrays and tensors in arithmetic; convert with torch.as_tensor(..., dtype=torch.double). Call .detach().cpu().numpy() before returning values to numpy code.
- Fit with `from botorch.fit import fit_gpytorch_mll` (NOT fit_gpytorch_model).
- SobolEngine takes `scramble=True`, NOT `randomized`: SobolEngine(dimension=dim, scramble=True, seed=seed).
- Raw objective values span several orders of magnitude, so consider log10(1 + y) plus standardization before fitting. Working in the unit cube and rescaling only when calling func also helps numerically.
- Set torch.manual_seed and np.random.seed from random_seed.
- Keep the algorithm cheap: use at most 5 optimizer restarts and 64 raw samples in optimize_acqf, because it runs many times.
- Do not wrap the whole loop in a try/except that falls back to random search.

Provide the Python code and a one-line description with the main idea (without enters). Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```
"""

BO_SKELETON_TASK_PROMPT = """Write a Python class Optimize_with_BO that minimizes a black-box function with Bayesian Optimization.
Below is a WORKING baseline. Improve it with ONE or TWO well-chosen changes that make it find good points faster. Keep the class name, the method names and the return format.

### Facts
- func is a MINIMIZATION problem: func(x) takes a numpy array of shape (dim,) and returns a float. Search box: func.bounds.lb to func.bounds.ub, here [-5, 5]^dim with dim = 6.
- Objective values are >= 0, span several orders of magnitude (0 to 10000+), and the landscape is multimodal.
- You may call func at most `budget` times in total (budget = 60, initial_points = 10). Results are scored by the area over the convergence curve, so early progress counts.

### Baseline (everything works as written)
```python
import numpy as np
import torch
from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import qLogExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood

class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int, random_seed: int):
        self.budget = budget
        self.dim = dim
        self.n_init = min(initial_points, budget)
        self.seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        # Create the sampler ONCE (re-creating it with the same seed repeats the same points).
        self.sobol = SobolEngine(dimension=dim, scramble=True, seed=random_seed)
        # Work in the unit cube [0, 1]^dim; rescale only when calling func.
        self.unit_bounds = torch.stack([torch.zeros(dim), torch.ones(dim)]).double()

    def _initial_design(self, n):
        # Returns a (n, dim) double tensor in [0, 1].
        # LHS alternative: torch.tensor(scipy.stats.qmc.LatinHypercube(d=self.dim, seed=self.seed).random(n))
        return self.sobol.draw(n).double()

    def _fit_model(self, X, y):
        # X: (n, dim) tensor in [0, 1]. y: (n,) tensor of RAW objective values (minimization).
        # BoTorch MAXIMIZES, so the target is log-transformed, standardized and NEGATED.
        z = torch.log10(1.0 + y.clamp_min(0.0))
        z = (z - z.mean()) / (z.std() + 1e-9)
        train_Y = (-z).unsqueeze(-1)  # shape (n, 1); this is the only place a dimension is added
        model = SingleTaskGP(X, train_Y)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        return model, train_Y.max()  # best value in the maximization convention

    def _next_point(self, model, best_f, X, y):
        # Returns a (1, dim) tensor in [0, 1].
        acq = qLogExpectedImprovement(model, best_f=best_f)
        cand, _ = optimize_acqf(acq, bounds=self.unit_bounds, q=1, num_restarts=5, raw_samples=64)
        return cand.detach()

    def _evaluate(self, func, x_unit, lb, ub):
        x = lb + (ub - lb) * x_unit.detach().cpu().numpy()
        return float(func(x))

    def __call__(self, func):
        lb = np.asarray(func.bounds.lb, dtype=float).reshape(-1)
        ub = np.asarray(func.bounds.ub, dtype=float).reshape(-1)
        X = self._initial_design(self.n_init)
        y = torch.tensor([self._evaluate(func, x, lb, ub) for x in X], dtype=torch.double)  # shape (n,)
        while len(y) < self.budget:
            model, best_f = self._fit_model(X, y)
            x_new = self._next_point(model, best_f, X, y)
            y_new = self._evaluate(func, x_new[0], lb, ub)
            X = torch.cat([X, x_new], dim=0)
            y = torch.cat([y, torch.tensor([y_new], dtype=torch.double)])
        i = int(torch.argmin(y))
        return float(y[i]), lb + (ub - lb) * X[i].numpy()
```

### Your task: make it better and different
Change the parts marked as free. Each change must be coherent, and the interfaces above must stay the same. Ideas:
- `_next_point`: a different acquisition (qUpperConfidenceBound with a decaying beta, Thompson sampling, a mix of two acquisitions), or a TRUST REGION: shrink or grow a radius r around the best point and pass `bounds = torch.stack([(xb - r).clamp(0, 1), (xb + r).clamp(0, 1)])` to optimize_acqf, where xb is the best x so far.
- `_fit_model`: another kernel via `SingleTaskGP(X, train_Y, covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim)))`, a different output transform, or a smaller or larger initial design.
- Adaptive rules (restart when stuck, switch from exploration to exploitation as the budget is used).

### Rules
- Use only torch, botorch, gpytorch, numpy and scipy.
- Keep the maximization convention: the model is fit on a target where larger is better, and best_f is its maximum.
- Keep X as a (n, dim) tensor and train_Y as a (n, 1) tensor. y stays 1-D (n,). Do not add extra dimensions.
- SingleTaskGP takes `covar_module=`, never `kernel=`. Do not pass arguments you are unsure exist.
- Never call func more than `budget` times. Never wrap the loop in a try/except that falls back to random search.
- Keep it cheap: num_restarts <= 5 and raw_samples <= 64 in optimize_acqf.

Provide the Python code and a one-line description with the main idea (without enters). Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```
"""


BO_TEMPLATE_TASK_PROMPT_UPDATED = """Design a novel and efficient Bayesian Optimization algorithm to minimize a black-box function. 
Early improvements are especially valuable because performance is scored by the area over the convergence curve.

Problem
Minimization problem: func(x) returns a scalar float, where x is a numpy array of shape (dim,).
Dimension: dim = 6.
Search bounds: [-5, 5]^dim, matching func.bounds.lb and func.bounds.ub.
Objective values are non-negative, heavy-tailed (approximately 0 to over 10000), and the landscape is multimodal.
Total evaluation budget: 60, including the initial design. Typically, initial_points = 10.
Never exceed the evaluation budget.
Code template

Keep the class name, method names, signatures, and return formats unchanged. Implement the missing methods and add any necessary hyperparameters with sensible defaults.

python
import numpy as np
import torch
from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel

class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int, random_seed: int):
        self.budget = budget
        self.dim = dim
        self.random_seed = random_seed
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])

        self.X = None
        self.y = None
        self.n_evals = 0
        self.n_init = min(initial_points, budget)

        # Add algorithm-specific hyperparameters with defaults.
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def _sample_points(self, n_points):
        \"\"\"Generate initial points using scrambled Sobol sampling.\"\"\"
        if n_points <= 0:
            return np.empty((0, self.dim))

        engine = SobolEngine(
            dimension=self.dim,
            scramble=True,
            seed=self.random_seed
        )
        unit_points = engine.draw(n_points).double().cpu().numpy()

        return self.bounds[0] + unit_points * (self.bounds[1] - self.bounds[0])

    def _fit_model(self, X, y):
        # Fit a GP surrogate using torch.double.
        # Handle heavy-tailed objective values using an appropriate
        # transformation, such as log10(1 + y), and standardization.
        # BoTorch acquisition functions maximize, so train on a target
        # that rewards smaller original objective values.
        # Return the fitted model.

    def _acquisition_function(self, X):
        # Evaluate the chosen acquisition function at X.
        # Return a numpy array of shape (n_points, 1).

    def _select_next_points(self, batch_size):
        # Select promising points using acquisition optimization,
        # candidate sampling, a trust region, or a hybrid strategy.
        # Return a numpy array of shape (batch_size, dim).
        # All points must lie within self.bounds.

    def _evaluate_points(self, func, X):
        # Evaluate func once for each row of X.
        # Never exceed self.budget - self.n_evals evaluations.
        # Update self.n_evals.
        # Return a numpy array of shape (n_points, 1).

    def _update_eval_points(self, new_X, new_y):
        # Append new observations to self.X and self.y.
        # Both must remain numpy arrays.

    def __call__(self, func):
        # 1. Generate and evaluate self.n_init Sobol initial points.
        # 2. Store the observations.
        # 3. Until the evaluation budget is exhausted:
        #       a. Fit the surrogate model.
        #       b. Select one new point.
        #       c. Evaluate it and update the observations.
        # 4. Return (best_f, best_x), where best_f is a float and
        #    best_x is a numpy array of shape (dim,).

        # IMPORTANT: Call func only inside _evaluate_points.

Algorithm design requirements

Choose ONE coherent and creative optimization strategy and implement it fully.

Possible ideas include:

Adaptive trust-region Bayesian Optimization.
Thompson sampling.
A portfolio of acquisition functions.
Adaptive exploration and exploitation.
A specialized kernel or output transformation.
A hybrid global and local search strategy.

Do not simply implement standard Expected Improvement with default GP settings. Explain the main idea in one short description.

Technical constraints (strict)
Use only NumPy, PyTorch, BoTorch, GPyTorch, and SciPy.
Use torch.double for all GP inputs, targets, bounds, and acquisition calculations.
Keep training shapes consistent:
train_X: (n, dim)
train_Y: (n, 1)
BoTorch maximizes. Ensure the surrogate target and acquisition function are consistent with minimizing the original objective. Correctly calculate the incumbent best_f in the transformed target space.
Use .detach().cpu().numpy() when converting tensors to NumPy.
Use fit_gpytorch_mll from botorch.fit.
Use scrambled Sobol sampling with SobolEngine(dimension=dim, scramble=True, seed=seed).
Normalize inputs to the unit cube [0, 1]^dim for GP fitting and acquisition optimization when appropriate. Rescale points to [-5, 5]^dim before evaluating func.
Handle heavy-tailed outputs carefully. Consider log10(1 + y) followed by standardization.
Keep computation cheap:
Maximum 5 optimizer restarts.
Maximum 64 raw samples in optimize_acqf.
Prefer optimizing one point per iteration.
Never evaluate more than the remaining budget.
Do not wrap the entire optimization loop in try/except or silently fall back to random search.
Handle numerical issues and GP fitting failures locally where appropriate, without hiding fundamental implementation errors.
Output requirements

Return only the following format:

Description: <one-line description of the main optimization idea>
Code:
python
<complete, executable Python implementation>


The implementation must include every method in the template, preserve all method signatures, and return the correct output format. """



NEW_PROMPT_1 = """
Description: <one-line explanation of the main optimization strategy>
Code:
python
<complete Python implementation>


You are an expert in Bayesian Optimization, Gaussian Processes, and efficient black-box global optimization. Your task is to implement a competitive Bayesian Optimization algorithm in Python.

Problem

Write a Python class named Optimize_with_BO with the following interface:

python
class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int, random_seed: int):
        ...

    def __call__(self, func):
        ...


The objective is to MINIMIZE an expensive, black-box function:

func(x) accepts a NumPy array of shape (dim,) and returns a scalar.
The search space is defined by func.bounds.lb and func.bounds.ub.
The default search space is [-5, 5]^dim.
The objective is potentially highly multimodal, with many local minima.
The dimensionality is typically 6.
The total evaluation budget is typically 60.
The initial design size is typically 10.
The random seed must be respected.

Your implementation must return:

python
best_f, best_x


where best_f is the smallest objective value evaluated and best_x is the corresponding NumPy array.

Optimization goal

The algorithm is evaluated using the Area Over the Convergence Curve (AOCC), or an equivalent normalized AUC-based convergence score.

The evaluation score is MAXIMIZED, with an ideal score of 1.0.

However, the underlying objective function is MINIMIZED.

This distinction is essential:

The objective is to find progressively smaller function values.
Finding a good solution early is more valuable than finding an equally good solution only at the end.
The algorithm should prioritize rapid early improvements while retaining sufficient exploration to avoid premature convergence.
The quality of the entire convergence trajectory matters, not just the final best objective value.

Design the algorithm specifically for this evaluation criterion.

Algorithm design freedom

Implement a genuine Bayesian Optimization algorithm using BoTorch and GPyTorch.

You are free to choose the architecture. Do not simply reproduce a conventional GP + Expected Improvement implementation unless you have a compelling reason.

You may consider, combine, or develop approaches involving:

Gaussian Process models with different covariance kernels.
Automatic relevance determination (ARD).
Expected Improvement, Log Expected Improvement, Upper Confidence Bound, or their noisy variants.
Thompson sampling.
Acquisition function portfolios or adaptive acquisition selection.
Trust-region Bayesian Optimization.
Adaptive exploration and exploitation.
Multiple candidate generation strategies.
Adaptive initial experimental designs.
Model-based global exploration and local refinement.
Restart mechanisms when optimization stagnates.
Input transformations or objective transformations.
Other methods supported by the BoTorch framework.

You do not need to implement all these ideas. Select a coherent strategy that is computationally efficient and appropriate for a small evaluation budget.

Avoid unnecessary complexity. A well-designed algorithm with one or two complementary strategies is preferable to a complicated collection of unrelated methods.

Technical requirements
Use only:
NumPy
PyTorch
BoTorch
GPyTorch
SciPy, optionally, for initial sampling or numerical utilities.
Do not use alternative optimization libraries, such as scikit-optimize, Optuna, Hyperopt, or external Bayesian Optimization implementations.
All objective evaluations must be performed through func(x).
Never exceed budget total objective evaluations. Initial design evaluations count toward the budget.
Use the provided random seed for reproducibility, including all applicable random samplers.
Support arbitrary positive dim, rather than hardcoding the dimensionality.
Normalize the input domain internally if beneficial, and correctly transform candidates back to the original bounds before evaluating the objective.
Maintain the best objective value and corresponding input throughout the entire optimization process.
The optimization loop must use a valid BoTorch-compatible probabilistic model and acquisition or sampling strategy.
Ensure that tensor dimensions, data types, model inputs, and acquisition function arguments are consistent with the selected BoTorch APIs.
Avoid unnecessary computational overhead. The evaluation budget is small, so model fitting and acquisition optimization should remain reasonably inexpensive.
Do not implement random search as the main optimization algorithm.
Do not include pseudocode, placeholders, incomplete methods, or undefined variables.
Do not use a try/except block that silently replaces Bayesian Optimization with random search.
Important implementation considerations
BoTorch acquisition functions generally use a MAXIMIZATION convention. If necessary, transform the minimization objective appropriately when training the surrogate model.
Keep track of raw objective values separately from transformed model targets.
Ensure the incumbent supplied to an acquisition function follows the maximization convention of that acquisition function.
Be careful when combining multiple acquisition functions with different output scales.
Use valid BoTorch APIs and imports. Do not invent function names, arguments, or methods.
The algorithm must work as a complete, standalone Python implementation.
Reference example: required interface and evaluation convention

The following is an example of the expected class structure. It is deliberately NOT a Bayesian Optimization algorithm. It demonstrates the evaluation budget, objective bounds, random seed, and return convention.

```python
import numpy as np

class RandomSearch:
    def __init__(self, budget: int, dim: int, initial_points: int, random_seed: int):
        self.budget = budget
        self.dim = dim
        self.initial_points = min(initial_points, budget)
        self.random_seed = random_seed

    def __call__(self, func):
        rng = np.random.default_rng(self.random_seed)

        lb = np.asarray(func.bounds.lb, dtype=float).reshape(-1)
        ub = np.asarray(func.bounds.ub, dtype=float).reshape(-1)

        best_f = np.inf
        best_x = None

        for _ in range(self.budget):
            x = rng.uniform(lb, ub)
            f = float(func(x))

            if f < best_f:
                best_f = f
                best_x = x.copy()

        return best_f, best_x
```
Final instruction

Design and implement your own competitive Bayesian Optimization algorithm. Do not simply copy the reference example or assume a particular acquisition function or GP architecture.

Return only the requested output format:

Description: <one-line description of the main algorithmic idea>
Code:
```python
<complete executable Python code>
```
"""


BO_STUB_TASK_PROMPT = """Design and implement a Bayesian Optimization class Optimize_with_BO to MINIMIZE a black-box function func(x) over [func.bounds.lb, func.bounds.ub] (default [-5, 5]^dim).
Problem & Goal
func(x) takes a 1D NumPy array of shape (dim,) and returns a scalar (float).
Typical setup: dim = 6, budget = 60, initial_points = 10, multimodal landscape (y >= 0, wide scale).
Total func(x) calls (including initial design) must NEVER exceed budget. Respect random_seed.
Scored by Area Over the Convergence Curve (AOCC): prioritize fast early drops while keeping enough exploration to avoid getting trapped in local minima.
Required Skeleton (keep class, method names, and return format; initial design is fixed to Sobol sampling)
import numpy as np
import torch
from torch.quasirandom import SobolEngine

```python
class Optimize_with_BO:
    def __init__(self, budget: int, dim: int, initial_points: int, random_seed: int):
        self.budget = budget
        self.dim = dim
        self.n_init = min(initial_points, budget)
        self.seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.sobol = SobolEngine(dimension=dim, scramble=True, seed=random_seed)
        self.unit_bounds = torch.stack([torch.zeros(dim), torch.ones(dim)]).double()

    def _initial_design(self, n: int) -> torch.Tensor:
        # Fixed: returns a (n, dim) double tensor in [0, 1]^dim using Sobol sampling.
        return self.sobol.draw(n).double()

    def _fit_model(self, X: torch.Tensor, y: torch.Tensor):
        # X: (n, dim) double tensor in [0, 1]; y: (n,) double tensor of RAW minimization values
        # Return (model, best_f_max) where larger target values are better
        ...

    def _next_point(self, model, best_f_max, X: torch.Tensor, y: torch.Tensor, step: int) -> torch.Tensor:
        # Return tensor of shape (1, dim) in [0, 1]
        ...

    def __call__(self, func):
        # Use self._initial_design(self.n_init), rescale [0, 1] -> [lb, ub] when calling func(x)
        # Return (best_f, best_x) with float best_f and shape-(dim,) NumPy array best_x
        ...
```
Design Freedom
Keep _initial_design fixed as Sobol sampling and implement a coherent, lightweight BO strategy for _fit_model, _next_point, and __call__. Vary your design across generations—for example:

Kernels & transforms: Matérn / RBF / ARD kernels, log or rank/quantile target transforms, trust regions, or periodic restarts on stagnation.
Acquisition strategy: qLogExpectedImprovement, qUpperConfidenceBound (with decaying or adaptive beta), qNoisyExpectedImprovement, portfolio/alternating acquisitions, or local + global candidate generation.
API & Shape Rules (Strictly Enforced)
Allowed libraries only: numpy, torch, botorch, gpytorch, and optionally scipy. No external BO libraries or silent fallback to random search.
Use dtype=torch.double (float64) for BoTorch tensors.
Keep _initial_design as Sobol sampling in [0, 1]^dim. Keep X as (n, dim), train_Y as (n, 1), and raw y as (n,). Unnormalize [0, 1] candidates to [lb, ub] before calling func(x).
BoTorch MAXIMIZES: negate/transform minimization targets so higher values are better, and pass best_f_max in that same maximization scale.
Valid imports & kwargs only:
Use from botorch.fit import fit_gpytorch_mll and fit_gpytorch_mll(mll) (NEVER fit_gpytorch_model).
If using a custom PyTorch/GPyTorch training loop, reduce loss to a scalar (.sum() or .mean()) before .backward().
Pass custom kernels to SingleTaskGP via covar_module=... (NEVER kernel=...).
Keep num_restarts <= 5 and raw_samples <= 64 in optimize_acqf.

Format your response strictly as:
Description: 
Code:
```python
<complete executable Python code>
```
"""


BO_STUB_TASK_PROMPT_1 = """
You are an expert in Bayesian Optimization and Python/BoTorch algorithm design. Your task is to write a complete, bug-free Python class implementing a Bayesian Optimization algorithm for continuous single-objective black-box minimization, evaluated using Area Over the Convergence Curve (AOCC) on multimodal benchmark functions (such as Rotated Rastrigin).



### Output Format (Strict)

Respond using ONLY the exact two-part structure below—no markdown fences outside the Code section, no extra explanations, and no conversational text:



Description: <A single concise sentence describing the surrogate model, target transformation, and acquisition strategy.>

Code:

```python

<Complete, self-contained Python implementation>

```



### Required Class Interface

Implement a class `Optimize_with_BO` with the following signature and behavior:

- `__init__(self, budget: int, dim: int, initial_points: int, random_seed: int)`

  - Store `budget`, `dim`, and `random_seed`.

  - Set `self.n_init = min(initial_points, budget)`.

- `__call__(self, func)`:

  - Minimize `func(x)` within `self.budget` total function evaluations.

  - Return `(best_f, best_x)` where `best_f` is a Python `float` representing the minimum objective value found and `best_x` is a 1D NumPy array of shape `(dim,)`.



### Critical Technical & API Rules (Must Follow Exactly)

1. **Objective Function Contract (`func`)**:

   - `func` may be an `ioh.iohcpp.problem.RealSingleObjective` instance or standard Python callable.

   - ALWAYS pass a 1D Python `list[float]` of length `dim` when calling `func`, and cast the output to `float`:

     `y_val = float(func(x_scaled_np.tolist()))`

   - NEVER pass a `torch.Tensor`, 2D list, or uncast return value directly to comparisons.

2. **Bounds Handling & Scaling**:

   - Extract bounds safely:

     ```python

     if hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):

         lb = np.asarray(func.bounds.lb, dtype=np.float64).reshape(-1)

         ub = np.asarray(func.bounds.ub, dtype=np.float64).reshape(-1)

         if lb.size == 1:

             lb = np.full(self.dim, lb.item(), dtype=np.float64)

         if ub.size == 1:

             ub = np.full(self.dim, ub.item(), dtype=np.float64)

     else:

         lb = np.full(self.dim, -5.0, dtype=np.float64)

         ub = np.full(self.dim, 5.0, dtype=np.float64)

     ```

   - Optimize internally on the unit hypercube `[0, 1]^dim` using `dtype=torch.float64`.

   - When using `botorch.utils.transforms.unnormalize(X, bounds)`, `bounds` MUST be a single `torch.Tensor` of shape `(2, self.dim)` and `dtype=torch.float64` (row 0 = lower bounds, row 1 = upper bounds). Never pass two separate 1D tensors.

   - To map unit points `x_unit` in `[0, 1]^dim` to domain coordinates:

     `x_scaled = lb + np.clip(x_unit, 0.0, 1.0) * (ub - lb)`

3. **Minimization-to-Maximization Consistency**:

   - BoTorch maximizes while `func` is minimized. Convert observed minimization values `y_tensor` into maximization targets `train_Y` of shape `(n, 1)` (for example, `y_neg = -y_tensor`, followed by standardization or monotonic rank/log compression).

   - `best_f_max` passed to acquisition functions (such as `qLogExpectedImprovement` or `ExpectedImprovement`) MUST equal `float(train_Y.max().item())` on that EXACT same transformed maximization scale. Never compute `-y.max()` on raw minimization values.

   - Use `from botorch.fit import fit_gpytorch_mll` and `fit_gpytorch_mll(mll)` (NEVER `fit_gpytorch_model`). Pass custom kernels to `SingleTaskGP` via `covar_module=...` (NEVER `kernel=...`).

4. **Budget & Numerical Stability**:

   - Use Sobol quasi-random sampling (`torch.quasirandom.SobolEngine`) for the initial design (`self.n_init` points).

   - Track `n_evals` strictly so total calls to `func` never exceed `self.budget`.

   - Keep `num_restarts <= 5` and `raw_samples <= 64` in `optimize_acqf`.

   - Wrap GP fitting (`fit_gpytorch_mll`) and `optimize_acqf` in a `try/except` fallback (e.g., perturbed best point or Sobol/local Gaussian candidate) so numerical Cholesky/L-BFGS errors never crash a run.

5. **Multimodal & AOCC Optimization Strategy**:

   - Prioritize rapid early improvement and escape from local optima using strategies such as:

     - Monotonic log/power or rank-based target warping to smooth steep multimodal ridges.

     - ARD Matérn-5/2 (`MaternKernel(nu=2.5, ard_num_dims=self.dim)`) or composite kernels.

     - `qLogExpectedImprovement` or `qUpperConfidenceBound` with decaying exploration weight `beta`, paired with `optimize_acqf` or local trust-region refinement when improvement stalls.

6. **Explicit Imports & Defined Symbols**:

     - Always import `MaternKernel` from `gpytorch.kernels` (`from gpytorch.kernels import MaternKernel`) whenever referenced.
     
     - Import `qLogExpectedImprovement` directly from `botorch.acquisition` (`from botorch.acquisition import qLogExpectedImprovement`), NOT from `botorch.acquisition.monte_carlo`.
     
     - Ensure every class, function, and module used anywhere in the code is explicitly imported at the top of the file—never rely on implicit or assumed imports.

7. **Acquisition & Fallback Safety**:

     - Do not pass `beta=...` to `qLogExpectedImprovement` unless verified for your installed BoTorch version (standard `qLogExpectedImprovement` takes `(model, best_f=...)`).
     
     - Inside `try/except` blocks for GP fitting or `optimize_acqf`, fall back to a valid candidate/point and continue the optimization loop rather than returning `(float('inf'), ...)` early.
     
     - Update GP training data cleanly (e.g., via `model.set_train_data(...)` or re-instantiating `SingleTaskGP` on the full `X` and `train_Y` tensors) with consistent tensor shapes and `dtype=torch.float64`.

### Evolutionary Refinement Instruction (Use When Mutating/Improving a Parent Candidate)

If a parent candidate and its traceback or fitness score are provided:

- **If `fitness == 0` or an exception occurred**: Diagnose and fix the exact root cause in the traceback first before altering acquisition hyperparameters. Do not blindly append unused keyword arguments to `optimize_acqf`.

- **If the parent ran cleanly (`fitness > 0`)**: Produce a meaningfully distinct algorithmic variation (e.g., changing target transformation, kernel structure, acquisition schedule, or adaptive trust-region/restart mechanism) rather than cosmetic parameter tweaks.
"""