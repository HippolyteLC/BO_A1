import os

import ioh

from . import ObjectiveFunction

        
class BBOBFunction(ObjectiveFunction):

    def __init__(self, dimension, log_dir=None):
        super().__init__(dimension, log_dir)
        self.problem = ioh.get_problem(
            15,                              
            instance=1,
            dimension=dimension,
            problem_class=ioh.ProblemClass.BBOB,
        )

    