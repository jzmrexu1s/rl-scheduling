from simso.core import Scheduler
from simso.schedulers import scheduler
from simso.core.Criticality import Criticality
from .EDF_VD_mono import EDF_VD_mono
from .ConfStaticEDFVD import static_optimal
from simso.core.Criticality import Criticality

@scheduler("simso.schedulers.EDF_VD_mono_CC")
class EDF_VD_mono_CC(EDF_VD_mono):
    def init(self):
        self.ready_list = []
        self.static_f_LO_LO, self.static_f_HI_LO, self.static_f_HI_HI, self.x = static_optimal(self.sim.task_list, 1, 0.2, 1, 2.5)
        for task in self.sim.task_list:
            task.deadline_offset = task.deadline * self.x - task.deadline

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
        job = None
        if self.ready_list:
            if self.sim.mode == Criticality.HI:
                # HI job with the highest priority
                ready_list_HI = [job for job in self.ready_list if job.task.criticality == Criticality.HI]
                if ready_list_HI:
                    job = min(ready_list_HI, key=lambda x: x.absolute_deadline)
            else:
                # job with the highest priority
                # print([job.absolute_deadline for job in self.ready_list])
                job = min(self.ready_list, key=lambda x: x.absolute_deadline)
        if job:
            self.sim.logger.log(str(self.sim.mode) + " Select " + job.name, kernel=True)
            self.set_speed_static(job)
            # print("cpu speed:", self.processors[0].speed)
        if not job:
            # self.sim.logger.log(str(self.sim.mode) + " Select None", kernel=True)
            if self.sim.mode == Criticality.HI:
                self.sim.handle_VD_reset()
                # self.sim.logger.log("Set mode to LO", kernel=True)
                return self.schedule(cpu)
        
        return (job, cpu)