import numpy as np
import sys
sys.path.append("..")
from simso.core.Criticality import Criticality
import openpyxl
from simso.utils.norm import norm

class Env(object):
    def __init__(self, model=None):
        self._model = model
        workbook = openpyxl.load_workbook('./mpc/output.xlsx')
        self.sheet = workbook['Run 5_ mpcACCsystem']
        self.column_to_id = {
            "a_ego": 2, "status": 3, "v_ego": 5, "v_lead": 6, "a_lead": 7, "safe_distance": 8
        }
        self.column_inteval = {
            "a_ego": 0.1, "status": 0.1, "v_ego": 0.01, "v_lead": 0.01, "a_lead": 0.01, "safe_distance": 0.01
        }
        
        
    def read_from_file(self, label, time):
        second = time / (1000000 * 1000)
        
        column = self.column_to_id[label]
        row = 2
        interval = self.column_inteval[label]
        
        if interval == 0.1:
            row += (int(second * 10) / 10) * 10
        if interval == 0.01:
            row += (int(second * 100) / 100) * 100
        return self.sheet.cell(row=int(row), column=column).value
        

    def observe_norm(self, time):
        # a_ego, v_ego, a_lead, v_lead, safe_distance
        a_ego = self.read_from_file("a_ego", time)
        v_ego = self.read_from_file("v_ego", time)
        a_lead = self.read_from_file("a_lead", time)
        v_lead = self.read_from_file("v_lead", time)
        safe_distance = self.read_from_file("safe_distance", time)
        # print(time, a_ego, v_ego, a_lead, v_lead, safe_distance)
        
        return np.array([
            norm(a_ego, -3, 2), norm(v_ego, 10, 40), 
            norm(a_lead, -1, 1), norm(v_lead, 10, 40), 
            norm(safe_distance, 30, 60)
        ])
    
    def now_acet(self, job, time):
        # ms
        if job.task.criticality == Criticality.HI:
            return self.read_from_file("status", time) * 2
        else:
            return job.wcet * 0.8
    
if __name__ == '__main__':
    env = Env()
    print(env.observe_norm(1668994370))