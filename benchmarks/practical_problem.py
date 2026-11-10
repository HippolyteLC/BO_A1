import os

import ioh
import numpy as np
from Practical_Problems import get_practical_problem

from . import ObjectiveFunction


class PracticalProblem(ObjectiveFunction):
    def __init__(self, dimension, log_dir):
        super().__init__(dimension, log_dir)
        self.problem = (
            get_practical_problem(1124, dim=dimension)
        )
