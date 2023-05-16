"""
Earliest Deadline First algorithm for uniprocessor architectures, supporting virtual deadline for MC tasks. 
"""
from simso.core import Scheduler
from simso.schedulers import scheduler
import time

@scheduler("simso.schedulers.EDF_VD_mono")
class EDF_VD_mono(Scheduler):
    def init(self):
        self.ready_list = []

    def on_activate(self, job):
        self.ready_list.append(job)
        job.acet = 5
        print(job.acet, '  ', time.time() )
        job.cpu.resched()

    def on_terminated(self, job):
        self.ready_list.remove(job)
        job.cpu.resched()

    def schedule(self, cpu):
        if self.ready_list:
            # job with the highest priority
            job = min(self.ready_list, key=lambda x: x.absolute_deadline + x.deadline_offset)
        else:
            job = None

        return (job, cpu)
