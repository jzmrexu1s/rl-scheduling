# coding=utf-8

from SimPy.Simulation import Simulation
from simso.core.Processor import Processor
from simso.core.Task import Task
from simso.core.Timer import Timer
from simso.core.etm import execution_time_models
from simso.core.Logger import Logger
from simso.core.results import Results
import numpy as np
from tools import *


class Model(Simulation):
    """
    Main class for the simulation. It instantiate the various components
    required by the simulation and run it.
    """

    def __init__(self, configuration, callback=None):
        """
        Args:
            - `callback`: A callback can be specified. This function will be \
                called to report the advance of the simulation (useful for a \
                progression bar).
            - `configuration`: The :class:`configuration \
                <simso.configuration.Configuration>` of the simulation.

        Methods:
        """
        Simulation.__init__(self)
        self._logger = Logger(self)
        task_info_list = configuration.task_info_list
        proc_info_list = configuration.proc_info_list
        self._cycles_per_ms = configuration.cycles_per_ms
        self.scheduler = configuration.scheduler_info.instantiate(self)
        self._energy_consumption = 0
        self._reward = 0

        try:
            self._etm = execution_time_models[configuration.etm](
                self, len(proc_info_list)
            )
        except KeyError:
            print("Unknowned Execution Time Model.", configuration.etm)

        self._task_list = []
        for task_info in task_info_list:
            self._task_list.append(Task(self, task_info))

        self._obs = self.calculate_obs()

        # Init the processor class. This will in particular reinit the
        # identifiers to 0.
        Processor.init()

        # Initialization of the caches
        for cache in configuration.caches_list:
            cache.init()

        self._processors = []
        for proc_info in proc_info_list:
            proc = Processor(self, proc_info)
            proc.caches = proc_info.caches
            self._processors.append(proc)

        # XXX: too specific.
        self.penalty_preemption = configuration.penalty_preemption
        self.penalty_migration = configuration.penalty_migration

        self._etm.init()

        self._duration = configuration.duration
        self.progress = Timer(self, Model._on_tick, (self,),
                              self.duration // 20 + 1, one_shot=False,
                              in_ms=False)
        self._callback = callback
        self.scheduler.task_list = self._task_list
        self.scheduler.processors = self._processors
        self.results = None

    def now_ms(self):
        return float(self.now()) / self._cycles_per_ms

    @property
    def logs(self):
        """
        All the logs from the :class:`Logger <simso.core.Logger.Logger>`.
        """
        return self._logger.logs

    @property
    def logger(self):
        return self._logger

    @property
    def cycles_per_ms(self):
        """
        Number of cycles per milliseconds. A cycle is the internal unit used
        by SimSo. However, the tasks are defined using milliseconds.
        """
        return self._cycles_per_ms

    @property
    def etm(self):
        """
        Execution Time Model
        """
        return self._etm

    @property
    def processors(self):
        """
        List of all the processors.
        """
        return self._processors

    @property
    def task_list(self):
        """
        List of all the tasks.
        """
        return self._task_list

    @property
    def duration(self):
        """
        Duration of the simulation.
        """
        return self._duration

    @property
    def energy_consumption(self):
        return self._energy_consumption

    @property
    def obs(self):
        return self._obs
    
    @property
    def reward(self):
        return self._reward

    def _on_tick(self):
        if self._callback:
            self._callback(self.now())

    def _calculate_jobs_count(self):
        count = 0
        for task in self.task_list:
            count += len(task.jobs)
        return count

    def set_reward(self, reward):
        self._reward += reward

    def reset_reward(self):
        self._reward = 0


    def run_model(self, load_rl=None):
        """ Execute the simulation."""
        self.initialize()
        self.scheduler.init()
        if load_rl is not None:
            self.load_model(load_rl)
        self.progress.start()

        for cpu in self._processors:
            self.activate(cpu, cpu.run())

        for task in self._task_list:
            self.activate(task, task.execute())

        try:
            self.simulate(until=self._duration)
        finally:
            self._etm.update()
            self.calculate_energy()

            if self.now() > 0:
                self.results = Results(self)
                self.results.end()

    def calculate_energy(self):
        for task in self._task_list:
            task.calculate_energy()
        self._energy_consumption = sum([item.energy_consumption for item in self._task_list])
        return self._energy_consumption

    def calculate_obs(self):
        wcets = [task.wcet for task in self._task_list]
        deadlines = [task.deadline for task in self._task_list]
        obs = []
        for task in self._task_list:
            for job in task.jobs:
                obs.append(
                    [
                        Norm(task.wcet, min(wcets), max(wcets)),
                        Norm(task.deadline, min(deadlines), max(deadlines)),
                        Norm(np.mean(wcets), min(wcets), max(wcets)),
                        Norm(np.mean(deadlines), min(deadlines), max(deadlines))
                    ]
                )
        self._obs = np.array(obs)

    def load_model(self, path):
        if self.scheduler.rl:
            self.scheduler.load_model(path)
    
    def save_model(self, path):
        if self.scheduler.rl:
            self.scheduler.save_model(path)