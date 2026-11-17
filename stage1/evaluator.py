# ____________________________________________________________________________
# Below Evaluator class code is adapted from the llamea github: 
# https://github.com/XAI-liacs/LLaMEA/tree/main/examples/
# ____________________________________________________________________________

from stage1.utils import aoc_logger, correct_aoc, OverBudgetException, convert_aucs

from ioh import logger

import numpy as np

from stage2.runner import make_problem

from stage1.utils import format_error


class Evaluator:
    def __init__(self, config, problem, testing=False, verbose=False):
        self.config = config
        self.problem = problem
        self.testing = testing
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

        fresh_env = {}
        try:
            exec(code, fresh_env)
            alg_class = fresh_env[algorithm_name]
        except KeyError as e:
            solution.set_scores(self.fall_back_score,
                feedback=f"Class '{algorithm_name}' was not found in your code. The class must be named Optimize_with_BO.")
            print(e)
            return solution
        except Exception as e:
            # If there's an error in code, set the score to 0
            error_feedback = format_error(e=e)
            solution.set_scores(self.fall_back_score, feedback=error_feedback)
            # solution.set_scores(self.fall_back_score, feedback=f"Runtime/Syntax error: {e}")
            print(e)
            return solution

        auc_mean = 0
        auc_std = 0
        aucs = []
        algorithm = None


        l2 = aoc_logger(budget=self.function_budget, upper=self.upper, lower=self.lower, triggers=[logger.trigger.ALWAYS])
        # problem = PracticalProblem(dimension=self.dimension).problem
        problem, f_opt = make_problem(self.problem)
        problem.attach_logger(l2)

        for seed in self.seeds:
            np.random.seed(seed)
            try:
                algorithm = alg_class(
                    budget=self.function_budget, dim=self.dimension, initial_points=self.initial_points, random_seed=seed
                )
                algorithm(problem)
            except OverBudgetException:
                pass
            except Exception as e:
                error_feedback = format_error(e=e)
                # solution.add_metadata("error", error_feedback)
                solution.set_scores(self.fall_back_score, error_feedback)
                return solution
            auc = correct_aoc(problem, l2, self.function_budget)
            aucs.append(auc)    
            l2.reset(problem)
            problem.reset()

            if self.verbose:
                print(f"AUC {auc} | seed {seed}")

            if self.testing: # only run a single seed during testing. 
                break

        aucs, auc_mean, auc_std = convert_aucs(aucs)

        feedback = f"The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.2f} with standard deviation {auc_std:0.2f}."

        print(algorithm_name, algorithm, auc_mean, auc_std)
        solution.add_metadata("aucs", aucs)
        solution.add_metadata("auc_mean", auc_mean)
        solution.add_metadata("auc_std", auc_std)    
        solution.set_scores(auc_mean, feedback)

        return solution
        
