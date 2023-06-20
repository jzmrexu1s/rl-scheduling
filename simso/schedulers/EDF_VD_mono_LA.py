import math
import numpy as np
from simso.core import Scheduler
from simso.schedulers import scheduler
from .EDF_VD_mono import EDF_VD_mono
from .ConfStaticEDFVD import static_optimal
from simso.core.Criticality import Criticality
from simso.core.Env import Env
from .EDF_VD_mono_LA_maxQoS import EDF_VD_mono_LA_maxQoS


@scheduler("simso.schedulers.EDF_VD_mono_LA")
class EDF_VD_mono_LA(EDF_VD_mono_LA_maxQoS):

    def set_speed_full(self):
        if self.sim.mode == Criticality.HI:
            self.processors[0].set_speed(1)
            return
        min_cycles, nearest_deadline, slack = self.slack()
        if nearest_deadline == self.sim.now_ms():
            self.processors[0].set_speed(1)
            self.sim.logger.log("Set speed 1", kernel=True)
        else:
            # print(min_cycles, nearest_deadline, self.sim.now_ms())
            accurate_speed = min_cycles / (nearest_deadline - self.sim.now_ms())
            self.processors[0].set_speed(min(1, math.ceil(100 * accurate_speed) / 100))
            self.sim.logger.log("Set speed " + str(min(1, math.ceil(100 * accurate_speed) / 100)), kernel=True)

    def on_pre_overrun(self, job):
        _, _, slack = self.slack()
        # Full speed when using slack on pre overrun. 
        self.processors[0].set_speed(1)
        # Not using slack to prevent overrun.
        job.set_pre_overrun_timer(0)

    def schedule(self, cpu):

        if self.sim.now() % 10 == 9 and self.sim.now() % 100 == 9:
            return (None, cpu)
        
        # select one job
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

            self.sim.logger.log(" Select " + job.name, kernel=True)
            # step in env afterwards
            self.set_speed_full()
            # print("cpu speed:", self.processors[0].speed)
        if not job:
            if self.sim.mode == Criticality.HI:
                self.sim.handle_VD_reset()
                return self.schedule(cpu)
            
            self.prev_state = self.state
        
        return (job, cpu)