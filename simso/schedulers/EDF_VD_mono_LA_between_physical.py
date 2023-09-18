import math
from simso.schedulers import scheduler
from simso.core.Criticality import Criticality
from .EDF_VD_mono_LA_between_base import EDF_VD_mono_LA_between_base
import random

@scheduler("simso.schedulers.EDF_VD_mono_LA_between_physical")
class EDF_VD_mono_LA_between_physical(EDF_VD_mono_LA_between_base):
    
    def get_by_name(self, name):
        env_state = self.sim.env.observe_norm(self.sim.now())
        names = ['a_ego', 'v_ego', 'a_lead', 'v_lead', 'safe_distance', 'headingAngle', 'lateralOffset', 'strength', 'departure_detected', 'lateral_deviation']
        for idx, _name in enumerate(names):
            if name == _name:
                return env_state[idx]
        return 0
        
    
    def get_speed_between(self, min_speed):
        
        
        if self.get_by_name('departure_detected') == 1:
            return min(min_speed + 0.5, 1)
        
        return min(min_speed + 0.2, 1)
