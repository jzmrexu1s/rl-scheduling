import math
import numpy as np
from simso.core import Scheduler
from simso.schedulers import scheduler
from .EDF_VD_mono import EDF_VD_mono
from .ConfStaticEDFVD import static_optimal
from simso.core.Criticality import Criticality
import rl.sac_classes as sac
import torch
import gym
from simso.core.Env import Env


DETERMINISTIC = False
batch_size  = 300
update_itr = 1
AUTO_ENTROPY = True
step = 500
@scheduler("simso.schedulers.EDF_VD_mono_LA_RL")
class EDF_VD_mono_LA_RL(EDF_VD_mono):

    def init(self):
        self.ready_list = []
        self.static_f_LO_LO, self.static_f_HI_LO, self.static_f_HI_HI, self.x = static_optimal(self.sim.task_list, 1, 0.2, 1, 2.5)
        for task in self.sim.task_list:
            if task.criticality == Criticality.HI:
                task.deadline_offset = task.deadline * self.x - task.deadline
        self.prev_state = None
        self.action = None
        self.prev_action = None
        self.prev_cycle = 0
        self.step = 0

    def on_pre_overrun(self, job):
        _, _, slack = self.slack()
        # 如果即将overrun，设置为最大速度
        self.processors[0].set_speed(1)
        # set pre overrun time using slack.
        job.set_pre_overrun_timer(max(0, slack))
        # no pre overrun
        # job.set_pre_overrun_timer(0)

    def U(self):
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
        return U

    def slack(self):
        U = self.U()
        if len(self.ready_list) == 0:
            return 0, 0, 0
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

    def set_speed_static(self, job):
        if self.sim.mode == Criticality.HI and job.task.criticality == Criticality.HI:
            self.processors[0].set_speed(self.static_f_HI_HI)
        elif self.sim.mode == Criticality.LO and job.task.criticality == Criticality.HI:
            self.processors[0].set_speed(self.static_f_HI_LO)
        else:
            self.processors[0].set_speed(self.static_f_LO_LO)

    def get_action(self, state):
        if self.frame_idx > self.explore_steps:
            action = self.sac_trainer.policy_net.get_action(state, deterministic = DETERMINISTIC)
        else:
            action = self.sac_trainer.policy_net.sample_action()
        return action

    def set_speed_rl(self, action):
        self.sim.logger.log("Set speed " + str(math.ceil(100 * (action + 0.5)) / 100), kernel=True)
        # TODO: 输出范围为-0.5~0.5，设置速度时直接加0.5
        self.processors[0].set_speed(math.ceil(100 * (action + 0.5)) / 100)

    def on_activate(self, job):
        self.ready_list.append(job)
        job.cpu.resched()

    def get_reward(self):
        abort_count = self.sim.etm.abort_count
        self.sim.etm.reset_count()
        energy_consumption = self.sim.speed_logger.default_multi_range_power(self.prev_cycle, self.sim.now())
        full_energy_consumption = self.sim.speed_logger.default_single_range_power(self.sim.now() - self.prev_cycle)
        if self.sim.now() - self.prev_cycle < 10:
            return -5 * abort_count + (full_energy_consumption - energy_consumption) / full_energy_consumption
        return -5 * abort_count * 1000000 / (self.sim.now() - self.prev_cycle) + (full_energy_consumption - energy_consumption) / full_energy_consumption

    def select_job(self):
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
        return job

    def schedule(self, cpu):
        self.step += 1
        if self.step >= step:
            self.sim.stopSimulation()
            return
        
        # init state
        if self.sim.now() == 0:
            job = self.select_job()
            _, _, slack = self.slack()
            state = np.append(np.array([job.task.wcet, job.ret, slack]), Env.observe(self, self.sim.now()))
            action = self.get_action(state)
            self.set_speed_rl(action)
            self.prev_state = state
            self.prev_action = action
            return (job, cpu)


        job = self.select_job()
        if job:
            self.sim.logger.log(" Select " + job.name, kernel=True)
            _, _, slack = self.slack()
            cur_state = np.append(np.array([job.task.wcet, job.ret, slack]), Env.observe(self, self.sim.now()))
            reward, done, _ = self.get_reward(), 0, {}
            print(self.prev_state, self.prev_action, reward, cur_state, self.sim.now())
            self.replay_buffer.push(self.prev_state, self.prev_action, reward, cur_state, done)
            self.episode_reward += reward
            self.frame_idx += 1
            if len(self.replay_buffer) > batch_size:
                for i in range(update_itr):
                    _=self.sac_trainer.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-1.*self.action_place.shape[0])
            
            action = self.get_action(cur_state)
            self.set_speed_rl(action)
            self.prev_state = cur_state
            self.prev_action = action

        else:
            if self.sim.mode == Criticality.HI:
                self.sim.handle_VD_reset()
                return self.schedule(cpu)
            

        return (job, cpu)