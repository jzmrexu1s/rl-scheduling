from .ACET import ACET
import random

class InjectACET(ACET):
    
    def on_activate(self, job):
        self.executed[job] = 0
        print(job.acet)
        self.et[job] = job.acet * self.sim.cycles_per_ms