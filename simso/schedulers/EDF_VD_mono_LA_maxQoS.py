import math

import numpy as np
from simso.core import Scheduler
from simso.schedulers import scheduler
from .EDF_VD_mono import EDF_VD_mono
from .ConfStaticEDFVD import static_optimal
from simso.core.Criticality import Criticality
from simso.core.Env import Env
import config

use_CC = config.use_CC
idle_speed = config.idle_speed
alpha = config.alpha

@scheduler("simso.schedulers.EDF_VD_mono_LA_maxQoS")
class EDF_VD_mono_LA_maxQoS(EDF_VD_mono):

    def init(self):
        self.ready_list = []
        self.static_f_LO_LO, self.static_f_HI_LO, self.static_f_HI_HI, self.x = static_optimal(self.sim.task_list, 1, idle_speed, 1, alpha)
        for task in self.sim.task_list:
            if task.criticality == Criticality.HI:
                task.deadline_offset = math.ceil(100 * (task.deadline * self.x - task.deadline)) / 100
        self.prev_state = None
        self.action = None
        self.prev_cycle = 0
        
        self.U_map_LO = {}
        


    def on_pre_overrun(self, job):
        _, _, slack = self.slack()
        # Full speed when using slack on pre overrun. 
        self.processors[0].set_speed(1)
        job.set_pre_overrun_timer(max(0, slack))
        
    def U(self):
        U = 0
        if self.sim.mode == Criticality.LO:
            for job in self.ready_list:
                if job.task.criticality == Criticality.LO:
                    U += (job.task.wcet / self.static_f_LO_LO) / job.task.period
                else:
                    U += (job.task.wcet / self.static_f_HI_LO) / (job.task.period + job.task.deadline_offset)
        else:
            for job in self.ready_list:
                if job.task.criticality == Criticality.HI:
                    U += (job.task.wcet_high / self.static_f_HI_HI) / job.task.period
        return U


    def slack(self):
        
        if use_CC:
            
            U = 0
            
            
            return
        
        
        U = self.U()
        
        if len(self.ready_list) == 0:
            return 0, 0, 0
        
        if self.sim.mode == Criticality.LO:
            ranked_jobs = sorted(self.ready_list, key=lambda x: x.absolute_deadline, reverse=True)
        else:
            ready_list_HI = [job for job in self.ready_list if job.task.criticality == Criticality.HI]
            ranked_jobs = sorted(ready_list_HI, key=lambda x: x.absolute_deadline, reverse=True)
        # print([[job.name, job.absolute_deadline] for job in ranked_jobs])
        nearest_deadline = ranked_jobs[-1].absolute_deadline
        p = 0
        for job in ranked_jobs:
            RC_i = job.ret
            if self.sim.mode == Criticality.LO:
                if job.task.criticality == Criticality.LO:
                    U -= job.task.wcet / job.task.period
                else:
                    U -= job.task.wcet / (job.task.period + job.task.deadline_offset)
            else:
                if job.task.criticality == Criticality.HI:
                    U -= job.task.wcet_high / job.task.period
            q_i = max(0, RC_i - (1 - U) * (job.absolute_deadline - nearest_deadline))
            if RC_i - q_i > 0:
                U = min(1.0, U + (RC_i - q_i) / (job.absolute_deadline - nearest_deadline))
            p = p + q_i
        return p, nearest_deadline, (nearest_deadline - self.sim.now_ms()) - p
        
        
    def set_speed_static(self, job):
        # print("f_LO_LO", self.static_f_LO_LO, "f_HI_LO", self.static_f_HI_LO, "f_HI_HI", self.static_f_HI_HI, "x", self.x)
        if self.sim.mode == Criticality.HI and job.task.criticality == Criticality.HI:
            self.processors[0].set_speed(self.static_f_HI_HI)
        elif self.sim.mode == Criticality.LO and job.task.criticality == Criticality.HI:
            self.processors[0].set_speed(self.static_f_HI_LO)
        else:
            self.processors[0].set_speed(self.static_f_LO_LO)

    def on_activate(self, job):
        self.ready_list.append(job)
        job.cpu.resched()

    def schedule(self, cpu):

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
            # print("cpu speed:", self.processors[0].speed)
        if not job:
            if self.sim.mode == Criticality.HI:
                self.sim.handle_VD_reset()
                return self.schedule(cpu)
            
            self.prev_state = self.state
        
        return (job, cpu)