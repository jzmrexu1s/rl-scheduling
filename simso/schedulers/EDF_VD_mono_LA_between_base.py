import math
import random
import numpy as np
from simso.core import Scheduler
from simso.schedulers import scheduler
from .ConfStaticEDFVD import static_optimal
from simso.core.Criticality import Criticality
from simso.core.Env import Env
from .EDF_VD_mono_LA import EDF_VD_mono_LA
import config

use_CC = config.use_CC


@scheduler("simso.schedulers.EDF_VD_mono_LA_between_base")
class EDF_VD_mono_LA_between_base(EDF_VD_mono_LA):
    
    def get_speed_between(self, min_speed):
        return min_speed
    
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
                speed_between = self.get_speed_between(accurate_speed)
                self.processors[0].set_speed(math.ceil(100 * speed_between) / 100)
                self.sim.logger.log("Set speed " + str(min(1, math.ceil(100 * speed_between) / 100)), kernel=True)

    def on_pre_overrun(self, job):
        _, _, slack = self.slack()
        # Full speed when using slack on pre overrun. 
        self.processors[0].set_speed(1)
        job.set_pre_overrun_timer(max(0, slack))