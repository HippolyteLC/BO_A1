import os

import ioh
import numpy as np
from Practical_Problems import get_practical_problem

from __init__ import ObjectiveFunction


class PracticalProblem(ObjectiveFunction):
    def __init__(self, dimension, seed, log_dir):
        super().__init__(dimension, seed, log_dir)
        self.problem = (
            get_practical_problem(1124, dim=dimension)
        )
