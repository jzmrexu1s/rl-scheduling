import math
from simso.schedulers import scheduler
from simso.core.Criticality import Criticality
from .EDF_VD_mono_LA_between_base import EDF_VD_mono_LA_between_base
import config
import random

@scheduler("simso.schedulers.EDF_VD_mono_LA_between_uni")
class EDF_VD_mono_LA_between_uni(EDF_VD_mono_LA_between_base):
    
    def get_speed_between(self, min_speed):
        return min(min_speed + 0.2, 1)
