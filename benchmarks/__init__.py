import ioh

import os

class ObjectiveFunction:
    def __init__(self, dimension, seed, log_dir):
        self.dimension = dimension
        self.seed = seed
        self.log_dir = log_dir
        self.problem = None

    def get_logger(self, algorithm_name, algorithm_info):
        logger = ioh.logger.Analyzer(
            triggers=[ioh.logger.trigger.OnImprovement(), ioh.logger.trigger.Each(10)],
            root=os.path.curdir,
            algorithm_name=algorithm_name,
            folder_name=self.log_dir,
            algorithm_info=algorithm_info,
            store_positions=False,
        )
        self.problem.attach_logger(logger)

    def __call__(self, solution):
        if self.problem is not None:
            return self.problem(solution)  
        return None