class Env(object):
    def __init__(self, model):
        self._model = model
    
    def now_acet(self, job):
        return min(job.wcet * self._model.now_ms() / 10, job.wcet)