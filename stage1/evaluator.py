
from benchmarks.practical_problem import PracticalProblem
from benchmarks.bbob import BBOBFunction

from stage1.utils import aoc_logger, correct_aoc, OverBudgetException, convert_aucs

from ioh import logger, get_problem

from config.config import ExperimentConfig, ModelConfig

from stage1.prompts import VANILLA_ROLE_PROMPT, VANILLA_TASK_PROMPT, ENGINEERED_ROLE_PROMPT, ENGINEERED_TASK_PROMPT


import numpy as np

from stage2.runner import make_problem

class Evaluator:
    def __init__(self, config, problem, verbose=False):
        self.config = config
        self.problem = problem
        self.verbose = verbose
        self.function_budget = config.llamea_function_budget
        self.dimension = config.dimension
        self.upper = config.upper
        self.lower = config.lower
        self.seeds = config.seeds
        self.initial_points = config.initial_points
        self.fall_back_score = config.fall_back

    def __call__(self, solution, explogger=None):
        code = solution.code
        algorithm_name = solution.name

        try:
            exec(code, globals())
        except Exception as e:
            # If there's an error in code, set the score to 0
            solution.set_scores(self.fallback_score, feedback=f"Runtime/Syntax error: {e}")
            print(e)
            return solution

        auc_mean = 0
        auc_std = 0
        aucs = []
        algorithm = None


        l2 = aoc_logger(budget=self.function_budget, upper=self.upper, lower=self.lower, triggers=[logger.trigger.ALWAYS])
        # problem = PracticalProblem(dimension=self.dimension).problem
        problem = make_problem(self.problem)
        problem.attach_logger(l2)

        for seed in self.seeds:
            np.random.seed(seed)
            try:
                algorithm = globals()[algorithm_name](
                    budget=self.function_budget, dim=self.dimension, initial_points=self.initial_points, random_seed=seed
                )
                algorithm(problem)
            except OverBudgetException:
                pass
            except Exception as e:
                error_feedback = (
                    f"Your generated algorithm crashed during execution.\n"
                    f"Exception: {type(e).__name__}\n"
                    f"Details: {str(e)}\n"
                    f"Please correct the implementation."
                )
                print(error_feedback)
                solution.add_metadata("error", error_feedback)
                solution.set_scores(self.fall_back_score, error_feedback)
                return solution
            auc = correct_aoc(problem, l2, self.function_budget)
            aucs.append(auc)    
            l2.reset(problem)
            problem.reset()

            if self.verbose:
                print(f"AUC {auc} | seed {seed}")

        aucs, auc_mean, auc_std = convert_aucs(aucs)

        feedback = f"The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.2f} with standard deviation {auc_std:0.2f}."

        print(algorithm_name, algorithm, auc_mean, auc_std)
        solution.add_metadata("aucs", aucs)
        solution.add_metadata("auc_mean", auc_mean)
        solution.add_metadata("auc_std", auc_std)    
        solution.set_scores(auc_mean, feedback)

        return solution
        
