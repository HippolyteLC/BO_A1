from algorithms.template import RandomSearch
from common import BudgetExceeded, CountedFunction, load_config, make_problem, set_all_seeds

cfg = load_config()

for name in ["bbob", "real"]:
    problem, f_opt = make_problem(name, cfg)
    set_all_seeds(120)

    func = CountedFunction(problem, budget=10)
    algo = RandomSearch(budget=20, dim=cfg["dimension"], random_seed=120)  # asks for 20 on purpose

    try:
        algo(func)
    except BudgetExceeded:
        print(name, ": stopped at budget")
        
    curve = []
    for v in func.history:
        curve.append(round(v, 2))

    print(name, ": f_opt =", f_opt, ", evals =", func.n_evals)
    print(name, ": best-so-far =", curve)
