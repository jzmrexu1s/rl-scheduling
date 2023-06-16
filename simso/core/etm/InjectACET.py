from .ACET import ACET

class InjectACET(ACET):

    def __init__(self, sim, _):
        self.sim = sim
        self.et = {}
        self.executed = {}
        self.on_execute_date = {}
        self.abort_count = 0
    
    def on_activate(self, job):
        self.executed[job] = 0
        job.acet = job.sim.env.now_acet(job)
        self.et[job] = job.acet * self.sim.cycles_per_ms

    def on_overrun(self, job):
        return super().update_executed(self)
    
    def on_abort(self, job):
        self.abort_count += 1
        self.update_executed(job)
        del self.et[job]

    def reset_count(self):
        self.abort_count = 0