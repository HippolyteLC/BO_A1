import cma
import numpy as np

from torch.quasirandom import SobolEngine


class CMAES:
    def __init__(self, budget, dim, initial_points, random_seed):
        self.budget = budget
        self.dim = dim
        self.initial_points = initial_points
        self.random_seed = random_seed
        self.lb = -5.0   # lower bound
        self.ub = 5.0    # upper bound
        self.sigma0 = 2.5  # initial step

    # ----- small helpers -----
 
    def evaluate(self, x):
        # Evaluate one point and remember the best so far
        f = self.func(x)
        self.n_evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = np.array(x)
        return f
 
    def budget_left(self):
        return self.budget - self.n_evals
 
    def random_point(self):
        return self.rng.uniform(self.lb, self.ub, size=self.dim)
 
    # ----- the two phases -----
 
    def initial_design(self):
     # randomly throw 10 points
        # for _ in range(self.initial_points):
        #     self.evaluate(self.random_point())
        sobol_engine = SobolEngine(dimension=self.dim, scramble=True, seed=self.random_seed)
        X_init = sobol_engine.draw(self.n_init).numpy()
        X_scaled = self.lb + X_init * (self.ub - self.lb)
        return X_scaled
    
    def run_one_cmaes(self, start_point, popsize, seed):
        # Run CMA-ES from start_point until it converges or the budget is used up
        options = {
            "bounds": [self.lb, self.ub],
            "popsize": popsize,
            "seed": seed,
            "verbose": -9, 
        }
        es = cma.CMAEvolutionStrategy(start_point, self.sigma0, options)
 
        while not es.stop():
            candidates = es.ask()   # CMA-ES proposes one generation of points
 
            if len(candidates) > self.budget_left():
                # Not enough budget for a full generation
                # evaluate the points we still can then stop
                for x in candidates[:self.budget_left()]:
                    self.evaluate(x)
                return
 
            values = []
            for x in candidates:
                values.append(self.evaluate(x))
            es.tell(candidates, values)   # CMA-ES updates its mean, step size and shape
 
    # ----- main -----
 
    def __call__(self, func):
        self.func = func
        self.rng = np.random.default_rng(self.random_seed)
        self.best_f = np.inf
        self.best_x = None
        self.n_evals = 0
        self.initial_design()
 
        # restarting until the budget is used up
        start_point = self.best_x
        popsize = 4 + int(3 * np.log(self.dim))   # default CMA-ES population size (9 for d = 6)
        restart = 0
        while self.budget_left() > 0:
            seed = self.random_seed * 100 + restart   # reproducible, different for every restart
            self.run_one_cmaes(start_point, popsize, seed)
 
            restart += 1
            popsize = popsize * 2
            start_point = self.random_point()
 
        return self.best_f, self.best_x