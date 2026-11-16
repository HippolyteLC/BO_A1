# Description: Role and Task Prompts for LLaMEA BO Algorithm Generation
# Used in the experimental pipeline

# ==============================================================================
# ROLE PROMPT
# ==============================================================================

ENGINEERED_ROLE_PROMPT = """You are an AI that generates Python optimization code by using Bayesian Optimization.
"""

# ==============================================================================
# TASK PROMPT 1 
# This prompt uses strict requirements to guide the LLM to make valid 
# code. It was created by iterating over prompts and finetuning the prompt to 
# fix tensor/np shape mismathcing, optimization direction confusion, 
# importerrors, etc. 
# 
# A very simple minimal code skeleton is provided, with some specific upper and lower
# bound guidance. 
# ==============================================================================

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

# ==============================================================================
# TASK PROMPT 2
# This prompt uses strict requirements to guide the LLM to make valid 
# code. 
# 
# A functional concrete BO skeleton is provided. 
# ==============================================================================



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