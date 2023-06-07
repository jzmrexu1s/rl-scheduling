from simso.core import Scheduler
from simso.schedulers import scheduler
from .EDF_VD_mono import EDF_VD_mono
from .ConfStaticEDFVD import static_optimal
from simso.core.Criticality import Criticality
import rl.sac_classes as sac
import torch
import gym

DETERMINISTIC = False
batch_size  = 300
update_itr = 1
AUTO_ENTROPY=True
@scheduler("simso.schedulers.EDF_VD_mono_CC")
class EDF_VD_mono_CC(EDF_VD_mono):

    def init(self):
        self.ready_list = []
        self.static_f_LO_LO, self.static_f_HI_LO, self.static_f_HI_HI, self.x = static_optimal(self.sim.task_list, 1, 0.2, 1, 2.5)
        for task in self.sim.task_list:
            task.deadline_offset = task.deadline * self.x - task.deadline

    def on_pre_overrun(self, job):
        _, _, slack = self.slack()
        # Full speed when using slack on pre overrun. 
        self.processors[0].set_speed(1)
        job.set_pre_overrun_timer(slack)
        # no pre overrun
        # job.set_pre_overrun_timer(0)

    def slack(self):
        U = 0
        if self.sim.mode == Criticality.LO:
            for job in self.ready_list:
                if job.task.criticality == Criticality.LO:
                    U += job.task.wcet / job.task.period
                else:
                    U += job.task.wcet / (job.task.period + job.task.deadline_offset)
        else:
            for job in self.ready_list:
                if job.task.criticality == Criticality.HI:
                    U += job.task.wcet_high / job.task.period
        ranked_jobs = sorted(self.ready_list, key=lambda x: x.absolute_deadline, reverse=True)
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
        self.processors[0].set_speed(p / (nearest_deadline - self.sim.now_ms()))
        
    def set_speed_full(self):
        min_cycles, nearest_deadline, slack = self.slack()
        self.processors[0].set_speed(min_cycles / (nearest_deadline - self.sim.now_ms()))

    def set_speed_static(self, job):
        # print("f_LO_LO", self.static_f_LO_LO, "f_HI_LO", self.static_f_HI_LO, "f_HI_HI", self.static_f_HI_HI, "x", self.x)
        if self.sim.mode == Criticality.HI and job.task.criticality == Criticality.HI:
            self.processors[0].set_speed(self.static_f_HI_HI)
        elif self.sim.mode == Criticality.LO and job.task.criticality == Criticality.HI:
            self.processors[0].set_speed(self.static_f_HI_LO)
        else:
            self.processors[0].set_speed(self.static_f_LO_LO)

    def set_speed(self):
        if self.frame_idx > self.explore_steps:
            action = self.sac_trainer.policy_net.get_action(self.state, deterministic = DETERMINISTIC)
        else:
            action = self.sac_trainer.policy_net.sample_action()
        next_state, reward, done, _ = self.env.step(action)
        self.replay_buffer.push(self.state, action, reward, next_state, done)
        self.state = next_state
        self.episode_reward += reward
        self.frame_idx += 1
        if len(self.replay_buffer) > batch_size:
            for i in range(update_itr):
                _=self.sac_trainer.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-1.*self.action_dim)


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
            self.set_speed_full()
            # print("cpu speed:", self.processors[0].speed)
        if not job:
            # self.sim.logger.log(str(self.sim.mode) + " Select None", kernel=True)
            if self.sim.mode == Criticality.HI:
                self.sim.handle_VD_reset()
                # self.sim.logger.log("Set mode to LO", kernel=True)
                return self.schedule(cpu)
        
        return (job, cpu)