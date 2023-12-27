import math
import time
import numpy as np
from simso.core import Scheduler
from simso.schedulers import scheduler
from .EDF_VD_mono import EDF_VD_mono
from .ConfStaticEDFVD import static_optimal
from simso.core.Criticality import Criticality
from simso.core.Env import Env
from .EDF_VD_mono_LA_maxQoS import EDF_VD_mono_LA_maxQoS
import config

use_CC = config.use_CC


@scheduler("simso.schedulers.EDF_VD_mono_LA")
class EDF_VD_mono_LA(EDF_VD_mono_LA_maxQoS):

    def set_speed(self):
        if self.sim.mode == Criticality.HI:
            self.processors[0].set_speed(1)
            return
        if use_CC:
            _, _, slack = self.slack()
        else:
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
        # Full speed when using slack on pre overrun. 
        self.processors[0].set_speed(1)
        # Not using slack to prevent overrun.
        job.set_pre_overrun_timer(0)

    def schedule(self, cpu):
        self.time_series.append(self.sim.now() / 1000000000)
        if (self.sim.now() == 0):
            self.power_series.append(0)
        else:
            self.power_series.append(self.sim.speed_logger.default_multi_range_power(0, self.sim.now()))
        if len(self.terminate_series) == 0:
            self.terminate_series.append(0)
        else:
            self.terminate_series.append(self.terminate_series[-1] + self.sim.etm.terminate_count)
        start_time = time.time()

        # if self.sim.now() % 10 == 9 and self.sim.now() % 100 == 9:
        #     return (None, cpu)
        
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
            self.set_speed()
            # print("cpu speed:", self.processors[0].speed)
        if not job:
            if self.sim.mode == Criticality.HI:
                self.sim.handle_VD_reset()
                return self.schedule(cpu)
            
            self.prev_state = self.state
            
        end_time = time.time()
        self.call_time += end_time - start_time
        self.call_count += 1
        self.sim.etm.reset_abort_count()
        self.sim.etm.reset_terminate_count()
        
        return (job, cpu)