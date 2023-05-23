"""
Earliest Deadline First algorithm for uniprocessor architectures, supporting virtual deadline for MC tasks. 
"""
from simso.core import Scheduler
from simso.schedulers import scheduler
from simso.core.Criticality import Criticality
import time

@scheduler("simso.schedulers.EDF_VD_mono")
class EDF_VD_mono(Scheduler):
    def init(self):
        self.ready_list = []

    def on_activate(self, job):
        self.ready_list.append(job)
        job.cpu.resched()

    def on_terminated(self, job):
        self.ready_list.remove(job)
        job.cpu.resched()

    def schedule(self, cpu):
        job = None
        if self.ready_list:
            if self.sim.mode == Criticality.HI:
                # HI job with the highest priority
                ready_list_HI = [job for job in self.ready_list if job.task.criticality == Criticality.HI]
                if ready_list_HI:
                    job = min(ready_list_HI, key=lambda x: x.absolute_deadline)
            else:
                # job with the highest priority
                job = min(self.ready_list, key=lambda x: x.absolute_deadline)
        if job:
            self.sim.logger.log(str(self.sim.mode) + " Select " + job.name, kernel=True)
        else:
            self.sim.logger.log(str(self.sim.mode) + " Select None", kernel=True)
            if self.sim.mode == Criticality.HI:
                self.sim.handle_reset_VD()
                self.sim.logger.log("Set mode to LO", kernel=True)
                self.schedule(cpu)
        return (job, cpu)
