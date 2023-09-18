from SimPy.Simulation import Monitor

import config

idle_speed = config.idle_speed
alpha = config.alpha

# idle_speed = 0.3
# alpha = 2.5

class SpeedLogger(object):

    def __init__(self, sim):
        self.sim = sim
        self._logs = Monitor(name="Logs", sim=sim)
        self._range_logs = []
        self.alpha = alpha
        self.beta = 1

    def log(self, status, job, speed, kernel=False):
        self._logs.observe(([status, job, speed], kernel))
        self.append_range_logs()

    def append_range_logs(self):
        new_log = self._logs[-1]
        cur_time = new_log[0]
        job = new_log[1][0][1]
        speed = new_log[1][0][2]
        prev_range_log = None
        if len(self._range_logs) > 0:
            prev_range_log = self._range_logs[-1]
        if new_log[1][0][0] == 'execute':
            if prev_range_log:
                prev_time = prev_range_log[0][1]
                if prev_range_log and cur_time > prev_time:
                    self._range_logs.append([[prev_time, cur_time], None, idle_speed])

            self._range_logs.append([[cur_time], job, speed])
        else:
            assert prev_range_log
            # assert prev_range_log[2] == speed
            # assert prev_range_log[1] == job
            prev_range_log[0] = [prev_range_log[0][0], cur_time]

    @property
    def logs(self):
        """
        The logs, a SimPy Monitor object.
        """
        return self._logs
    
    @property
    def range_logs(self):
        return self._range_logs
    
    def default_single_range_power(self, time_gap):
        return self.single_range_power(self.alpha, self.beta, 1, time_gap)
    
    def default_multi_range_power(self, start_time, end_time):
        return self.multi_range_power(self.alpha, self.beta, start_time, end_time)
    
    def multi_range_power(self, alpha, beta, start_time, end_time):
        power = 0
        final_start_time = None
        # print(self._range_logs)
        if len(self._range_logs[-1][0]) == 1:
            final_start_time = self._range_logs[-1][0][0]
            if end_time >= final_start_time:
                self._range_logs[-1][0] = [final_start_time, end_time]
            else:
                self._range_logs[-1][0] = [final_start_time, final_start_time]
        p = len(self._range_logs) - 1

        while end_time <= self._range_logs[p][0][0]:
            p -= 1

        while p >= 0 and start_time <= self._range_logs[p][0][1] and start_time <= self._range_logs[p][0][0]:
            speed = self._range_logs[p][2]
            if end_time >= self._range_logs[p][0][0] and end_time <= self._range_logs[p][0][1]:
                # print(self.single_range_power(alpha, beta, speed, end_time - self._range_logs[p][0][0]))
                power += self.single_range_power(alpha, beta, speed, end_time - self._range_logs[p][0][0])
            elif start_time >= self._range_logs[p][0][0] and end_time <= self._range_logs[p][0][1]:
                # print(self.single_range_power(alpha, beta, speed, self._range_logs[p][0][1] - start_time))
                power += self.single_range_power(alpha, beta, speed, self._range_logs[p][0][1] - start_time)
            else:
                # print(self.single_range_power(alpha, beta, speed, self._range_logs[p][0][1] - self._range_logs[p][0][0]))
                power += self.single_range_power(alpha, beta, speed, self._range_logs[p][0][1] - self._range_logs[p][0][0])
            p -= 1
        if final_start_time:
            self._range_logs[-1][0] = [final_start_time]
        return power

    def single_range_power(self, alpha, beta, speed, time):
        return self.power(alpha, beta, speed) * time / self.sim.cycles_per_ms
    
    def power(self, alpha, beta, speed):
        return beta * pow(speed, alpha)
    
    
    
