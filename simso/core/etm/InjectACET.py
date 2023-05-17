from .ACET import ACET

class InjectACET(ACET):
    
    def on_activate(self, job):
        self.executed[job] = 0
        job.acet = job.sim.env.now_acet(job)
        self.et[job] = job.acet * self.sim.cycles_per_ms