# dummy run on the skeleton to check if evaluate function from bo_gen works. 

from stage1.bo_gen import evaluate


class DummySolution:
    def __init__(self, code, name):
        self.code, self.name, self.metadata = code, name, {}
    def set_scores(self, fitness, feedback=""):
        self.fitness, self.feedback = fitness, feedback
    def add_metadata(self, key, value):
        self.metadata[key] = value

sol = evaluate(DummySolution(open("stage1/skeleton.py").read(), "Optimize_with_BO"))
print(sol.fitness, sol.feedback)
print(sol.metadata)