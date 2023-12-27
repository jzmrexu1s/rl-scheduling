import math

import numpy as np
from simso.core import Scheduler
from simso.schedulers import scheduler
from .EDF_VD_mono_LA_maxQoS import EDF_VD_mono_LA_maxQoS
from .ConfStaticEDFVD import static_optimal
from simso.core.Criticality import Criticality
from simso.core.Env import Env
import config

@scheduler("simso.schedulers.EDF_VD_mono_new")
class EDF_VD_mono_new(EDF_VD_mono_LA_maxQoS):
    
    def on_pre_overrun(self, job):
        self.processors[0].set_speed(1)
        job.set_pre_overrun_timer(0)