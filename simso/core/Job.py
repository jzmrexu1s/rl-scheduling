# coding=utf-8

import math
from SimPy.Simulation import Process, hold, passivate
from simso.core.JobEvent import JobEvent
from math import ceil
from simso.core.Timer import Timer
from simso.core.Criticality import Criticality

"""
MC behavior refers to: 
The preemptive uniprocessor scheduling of mixed-criticality implicit-deadline sporadic task systems
"""

class Job(Process):
    """The Job class simulate the behavior of a real Job. This *should* only be
    instantiated by a Task."""

    def __init__(self, task, name, pred, monitor, etm, sim, acet=None):
        """
        Args:
            - `task`: The parent :class:`task <simso.core.Task.Task>`.
            - `name`: The name for this job.
            - `pred`: If the task is not periodic, pred is the job that \
            released this one.
            - `monitor`: A monitor is an object that log in time.
            - `etm`: The execution time model.
            - `sim`: :class:`Model <simso.core.Model>` instance.

        :type task: GenericTask
        :type name: str
        :type pred: bool
        :type monitor: Monitor
        :type etm: AbstractExecutionTimeModel
        :type sim: Model
        """
        Process.__init__(self, name=name, sim=sim)
        self._task = task
        self._pred = pred
        self.instr_count = 0  # Updated by the cache model.
        self._computation_time = 0
        self._last_exec = None
        self._n_instr = task.n_instr
        self._start_date = None
        self._end_date = None
        self._is_preempted = False
        self._activation_date = self.sim.now_ms()
        self._absolute_deadline = self.sim.now_ms() + task.deadline
        self._aborted = False
        self._sim = sim
        self._monitor = monitor
        self._etm = etm
        self._was_running_on = task.cpu
        self._acet = acet if acet else self.wcet
        self._is_pre_overrun = False

        self._on_activate()

        self.context_ok = True  # The context is ready to be loaded.

        self.timer_overrun = None

    def is_active(self):
        """
        Return True if the job is still active.
        """
        return self._end_date is None
    
    def renew_deadline_VD_overrun(self):
        # print(self.name, self._absolute_deadline, self._task.deadline_offset)
        self._absolute_deadline = self._absolute_deadline - self._task.deadline_offset

    def renew_deadline_VD_reset(self):
        self._absolute_deadline = self._absolute_deadline + self._task.deadline_offset

    def set_pre_overrun_timer(self, slack):
        ret = self._etm.get_ret(self)
        self._sim.logger.log(self.name + "   " + str(ret) + " " + str(self._etm.get_ret(self)))
        if ret >= 0:
            self._sim.logger.log(self.name + " About to Overrun! Current computation time: " + str(self.computation_time * self.sim.cycles_per_ms) + " ret: " + str(self._etm.get_ret(self)) + " using slack: " + str(slack * self.sim.cycles_per_ms), kernel=True)
            # print("Overrun timer: ", self.name, self.absolute_deadline, self.ret)
            self._sim.logger.log("Pre overrun timer: " + self.name + " " + str(self.sim.now_ms() + slack) + " " + str(ret))
            self.timer_overrun = Timer(self.sim, self._on_overrun,
                                (), slack)
            self.timer_overrun.start()

    def _on_pre_overrun(self):
        ret = self._etm.get_ret(self)
        self._sim.logger.log(self.name + " preoverrun, ret " + str(ret))
        self._is_pre_overrun = True
        self._task.cpu.pre_overrun(self)
        self._etm.on_pre_overrun(self)

    def _on_overrun(self):
        ret = self._etm.get_ret(self)
        # if ret >= 0:
            # print(self.name + " Overrun! Current computation time: " + str(self.computation_time) + " ret: " + str(ret))
        self._sim.logger.log(self.name + " Overrun! Current computation time: " + str(self.computation_time) + " ret: " + str(ret), kernel=True)
        self._sim.handle_VD_overrun()
        self._task.cpu.overrun(self)
        self._etm.on_overrun(self)

    def _on_simple_overrun(self):
        ret = self._etm.get_ret(self)
        # if ret >= 0:
            # print(self.name + " Overrun! Current computation time: " + str(self.computation_time) + " ret: " + str(ret))
        self._sim.logger.log(self.name + " Overrun! Current computation time: " + str(self.computation_time) + " ret: " + str(ret), kernel=True)
        self.abort()
        
    def _on_activate(self):
        self._monitor.observe(JobEvent(self, JobEvent.ACTIVATE))
        self._sim.logger.log(self.name + " Activated.", kernel=True)
        self._etm.on_activate(self)

    def _on_execute(self):
        self._last_exec = self.sim.now()
        

        self._etm.on_execute(self)
        if self._is_preempted:
            self._is_preempted = False

        self.cpu.was_running = self

        self._monitor.observe(JobEvent(self, JobEvent.EXECUTE, self.cpu))
        # if self.timer_overrun: print(self.timer_overrun.delay)
        # Timer has already added, if in pre_overrun status
        if self.sim.mc and self.sim.mode == Criticality.LO and not self._is_pre_overrun and self.task.criticality == Criticality.HI:
            if self.ret >= 0:
                left_time = self.ret / self.cpu.speed
                self._sim.logger.log("HI task in LO mode overrun timer: " + self.name + " " + str(self.sim.now_ms() + math.ceil(100 * left_time) / 100) + " " + str(self.ret))
                self.timer_overrun = Timer(self.sim, self._on_pre_overrun,
                                (), math.ceil(100 * left_time) / 100)
                self.timer_overrun.start()
        elif self.sim.mc and self.sim.mode == Criticality.LO and not self._is_pre_overrun:
            if self.ret >= 0:
                left_time = self.ret / self.cpu.speed
                # print(self.sim.now_ms(), math.ceil(100 * left_time) / 100)
                self._sim.logger.log("LO task overrun timer: " + self.name + " " + str(self.sim.now_ms() + math.ceil(100 * left_time) / 100) + " " + str(self.ret) + " cpu speed: " + str(self.cpu.speed))
                self.timer_overrun = Timer(self.sim, self._on_simple_overrun,
                                (), math.ceil(100 * left_time) / 100)
                self.timer_overrun.start()
        elif self.sim.mc and self.sim.mode == Criticality.HI:
            if self.ret >= 0:
                left_time = self.ret / self.cpu.speed
                self._sim.logger.log("HI task in HI mode overrun timer: " + self.name + " " + str(self.sim.now_ms() + math.ceil(100 * left_time) / 100) + " " + str(self.ret) + " cpu speed: " + str(self.cpu.speed))
                
                self.timer_overrun = Timer(self.sim, self._on_simple_overrun,
                                (), math.ceil(100 * left_time) / 100)
                self.timer_overrun.start()
        self._sim.logger.log("{} Executing on {}".format(
            self.name, self._task.cpu.name), kernel=True)
        self._sim.speed_logger.log("execute", self, self._sim.scheduler.processors[0].speed, kernel=True)

    def _on_stop_exec(self):
        if self.sim.mc and self.timer_overrun:
            self.timer_overrun.stop()
        if self._last_exec is not None:
            self._computation_time += self.sim.now() - self._last_exec
        self._last_exec = None
        self._sim.speed_logger.log("stop", self, self._sim.scheduler.processors[0].speed, kernel=True)

    def _on_preempted(self):
        self._on_stop_exec()
        self._etm.on_preempted(self)
        self._is_preempted = True
        self._was_running_on = self.cpu

        self._monitor.observe(JobEvent(self, JobEvent.PREEMPTED))
        self._sim.logger.log(self.name + " Preempted! ret: " +
                             str(self.interruptLeft), kernel=True)

    def _on_terminated(self):
        self._on_stop_exec()
        self._etm.on_terminated(self)

        self._end_date = self.sim.now()
        self._monitor.observe(JobEvent(self, JobEvent.TERMINATED))
        self._task.end_job(self)
        self._task.cpu.terminate(self)
        self._sim.logger.log(self.name + " Terminated.", kernel=True)

    def _on_abort(self):
        self._on_stop_exec()
        self._etm.on_abort(self)
        self._end_date = self.sim.now()
        self._aborted = True
        self._monitor.observe(JobEvent(self, JobEvent.ABORTED))
        self._task.end_job(self)
        self._task.cpu.terminate(self)
        self._sim.logger.log("Job " + str(self.name) + " aborted! ret:" + str(self.ret))

    def is_running(self):
        """
        Return True if the job is currently running on a processor.
        Equivalent to ``self.cpu.running == self``.

        :rtype: bool
        """
        return self.cpu.running == self

    def abort(self):
        """
        Abort this job. Warning, this is currently only used by the Task when
        the job exceeds its deadline. It has not be tested from outside, such
        as from the scheduler.
        """
        self._on_abort()

    @property
    def aborted(self):
        """
        True if the job has been aborted.

        :rtype: bool
        """
        return self._aborted

    @property
    def exceeded_deadline(self):
        """
        True if the end_date is greater than the deadline or if the job was
        aborted.
        """
        return (self._absolute_deadline * self._sim.cycles_per_ms <
                self._end_date or self._aborted)

    @property
    def start_date(self):
        """
        Date (in ms) when this job started executing
        (different than the activation).
        """
        return self._start_date

    @property
    def end_date(self):
        """
        Date (in ms) when this job finished its execution.
        """
        return self._end_date

    @property
    def response_time(self):
        if self._end_date:
            return (float(self._end_date) / self._sim.cycles_per_ms -
                    self._activation_date)
        else:
            return None

    @property
    def ret(self):
        """
        Remaining execution time in ms.
        """
        if self._is_pre_overrun: return self.task.wcet_high - self.actual_computation_time
        return self.wcet - self.actual_computation_time

    @property
    def ret_cycle(self):
        return self._etm.get_ret(self)

    @property
    def laxity(self):
        """
        Dynamic laxity of the job in ms.
        """
        return (self.absolute_deadline - self.ret
                ) * self.sim.cycles_per_ms - self.sim.now()

    @property
    def computation_time(self):
        """
        Time spent executing the job in ms.
        """
        return float(self.computation_time_cycles) / self._sim.cycles_per_ms

    @property
    def computation_time_cycles(self):
        """
        Time spent executing the job.
        """
        if self._last_exec is None:
            return int(self._computation_time)
        else:
            return (int(self._computation_time) +
                    self.sim.now() - self._last_exec)

    @property
    def actual_computation_time(self):
        """
        Computation time in ms as if the processor speed was 1.0 during the
        whole execution.
        """
        return float(
            self.actual_computation_time_cycles) / self._sim.cycles_per_ms

    @property
    def actual_computation_time_cycles(self):
        """
        Computation time as if the processor speed was 1.0 during the whole
        execution.
        """
        return self._etm.get_executed(self)

    @property
    def cpu(self):
        """
        The :class:`processor <simso.core.Processor.Processor>` on which the
        job is attached. Equivalent to ``self.task.cpu``.
        """
        return self._task.cpu

    @property
    def task(self):
        """The :class:`task <simso.core.Task.Task>` for this job."""
        return self._task

    @property
    def data(self):
        """
        The extra data specified for the task. Equivalent to
        ``self.task.data``.
        """
        return self._task.data

    @property
    def wcet(self):
        """
        Worst-Case Execution Time in milliseconds.
        Equivalent to ``self.task.wcet``.
        """
        if self._sim.mode == Criticality.LO:
            return self._task.wcet
        return self._task.wcet_high

    @property
    def activation_date(self):
        """
        Activation date in milliseconds for this job.
        """
        return self._activation_date

    @property
    def absolute_deadline(self):
        """
        Absolute deadline in milliseconds for this job. This is the activation
        date + the relative deadline.
        """
        return self._absolute_deadline

    @property
    def absolute_deadline_cycles(self):
        return self._absolute_deadline * self._sim.cycles_per_ms

    @property
    def period(self):
        """Period in milliseconds. Equivalent to ``self.task.period``."""
        return self._task.period

    @property
    def deadline(self):
        """
        Relative deadline in milliseconds.
        Equivalent to ``self.task.deadline``.
        """
        return self._task.deadline

    @property
    def pred(self):
        return self._pred
    
    @property
    def acet(self):
        return self._acet
    
    @acet.setter
    def acet(self, value):
        self._acet = value

    @property
    def etm(self):
        return self._etm

    def activate_job(self):
        self._start_date = self.sim.now()
        # Notify the OS.
        self._task.cpu.activate(self)

        # While the job's execution is not finished.
        while self._end_date is None:
            # Wait an execute order.
            yield passivate, self

            # Execute the job.
            if not self.interrupted():
                self._on_execute()
                # ret is a duration lower than the remaining execution time.
                ret = self._etm.get_ret(self)

                while ret > 0:
                    yield hold, self, int(ceil(ret))

                    if not self.interrupted():
                        # If executed without interruption for ret cycles.
                        ret = self._etm.get_ret(self)
                    else:
                        self._on_preempted()
                        self.interruptReset()
                        break

                if ret <= 0:
                    # End of job.
                    self._on_terminated()

            else:
                self.interruptReset()
