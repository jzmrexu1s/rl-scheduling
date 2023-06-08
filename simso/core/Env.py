import numpy as np
from simso.core.Criticality import Criticality

class Env(object):
    def __init__(self, model):
        self._model = model

    def observe(self, time):
        # time: ms
        return np.array([1, 1, 1, 1])
    
    def now_acet(self, job):
        # return min(job.wcet * self._model.now_ms() / 10, job.wcet)
        if job.task.criticality == Criticality.HI:
            return job.wcet * 1.2
        return job.wcet