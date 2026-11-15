# dummy run on the skeleton to check if evaluate function from bo_gen works. 

from stage1.evaluator import Evaluator

from config.config import ExperimentConfig

class DummySolution:
    def __init__(self, code, name):
        self.code, self.name, self.metadata = code, name, {}
    def set_scores(self, fitness, feedback=""):
        self.fitness, self.feedback = fitness, feedback
    def add_metadata(self, key, value):
        self.metadata[key] = value

sol = Evaluator(ExperimentConfig(), "real")(DummySolution(open("stage1/test_2.py").read(), "Optimize_with_BO"))
print(sol.fitness, sol.feedback)
print(sol.metadata)