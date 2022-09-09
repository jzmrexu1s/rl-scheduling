from audioop import reverse
from tkinter.messagebox import NO
from simso.core import Scheduler
from simso.schedulers import scheduler
from MADDPG.MADDPG import MADDPG as MADDPG_PROCESS
import numpy as np
import torch as th
from MADDPG.params import scale_reward
import copy
from simso.core.tools import *


@scheduler("simso.schedulers.MADDPG")
class MADDPG(Scheduler):
    

    def init(self):
        np.random.seed(1234)
        th.manual_seed(1234)
        # world.seed(1234)
        n_agents = 4
        n_states = 4
        n_actions = 8
        capacity = 1000000
        batch_size = 1000

        episodes_before_train = 100

        win = None
        param = None

        self.action = None
        self.step = 0
        self.max_steps = 1000

        self.maddpg = MADDPG_PROCESS(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)
        self.FloatTensor = th.cuda.FloatTensor if self.maddpg.use_cuda else th.FloatTensor

    def load_model(self, path):
        self.maddpg.load_model(path)

    def save_model(self, path):
        self.maddpg.save_model(path)

    def on_activate(self, job):
        # print('on_activate: ', job.name)
        job.cpu.resched()

    def on_terminated(self, job):
        # print('on_terminated:  ', job.name)
        job.cpu.resched()

    def select_by_action(self, action):
        decisions = []
        jobs = {}
        for task in self.task_list:
            if task.job.is_active():
                jobs[task.identifier] = task.job
        for (i, item) in enumerate(action):
            _item = [(x, i) for (i, x) in enumerate(item)] 
            _item.sort(reverse=True, key=lambda x: x[0])
            j = 0
            if decisions:
                while j < len(_item) and (_item[j][1] in [x[0].task.identifier for x in decisions] or _item[j][1] not in [item.task.identifier for item in jobs.values()]):
                    j += 1
            if j < len(_item) and _item[j][1] in [item.task.identifier for item in jobs.values()]:
                decisions.append((jobs[_item[j][1]], self.processors[i]))
        return decisions


    def schedule(self, cpu):
        if self.maddpg.steps_done > self.max_steps:
            raise MaxStepException
        prev_obs = copy.copy(self.sim.obs)
        self.sim.calculate_obs()
        obs = self.FloatTensor(self.sim.obs)
        if prev_obs is not None:
            self.maddpg.memory.push(prev_obs.data, self.action, obs, self.sim.reward)
            c_loss, a_loss = self.maddpg.update_policy()

        action = self.maddpg.select_action(obs).data.cpu()
        self.action = action
        self.step += 1
        self.sim.reset_reward()
        return self.select_by_action(action)