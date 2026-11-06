import os

import ioh

from __init__ import ObjectiveFunction

        
class BBOBFunction(ObjectiveFunction):
    def __init__(self, dimension, seed, log_dir):
        super().__init__(dimension, seed, log_dir)
        self.problem = ioh.get_problem(
            15,                              
            instance=1,
            dimension=dimension,
            problem_class=ioh.ProblemClass.BBOB,
        )

    