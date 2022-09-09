import sys
sys.path.append("../simso")
from simso.configuration import Configuration
from simso.utils import PartitionedScheduler
from simso.core.Scheduler import SchedulerInfo, PartitionFailedException
from simso.core.tools import *
from simso.core import Model
from simso.generator.task_generator import gen_uunifastdiscard, gen_randfixedsum, gen_periods_uniform, gen_tasksets
import numpy as np
import math
from scipy.stats import loguniform
from config import *

rl_schedulers_class = ['simso.schedulers.MADDPG']

def run_single(model, load=False):
    try:
        if load:
            model.run_model(load_rl=MODEL_DIR)
        else:
            model.run_model()
        model.save_model(MODEL_DIR)
    except MaxStepException:
        return True, model
    except PartitionFailedException:
            print('Partition Failed! ')
            return False, model
    except AbortException as e:
        print('Job Aborted! ')
        return False, model
    return True, model

def export_log(model, name):
    with open (LOG_DIR + '/log_' + name + '.log', 'w') as f:
        for log in model.logs:
            f.writelines(str(log) + '\n')
        if model.results:
            for task in model.results.tasks:
                f.writelines(task.name + "  period: " + str(task.period) + " wcet: " + str(task.wcet) + " energy_consumption: " + str(task.energy_consumption) + '\n')
                for job in task.jobs:
                    f.writelines("cpu_speed:" + str(job.cpu_speed) +
                     " computation_time:" + str(job.computation_time) + 
                     " computation_time_cycles: " + str(job.computation_time_cycles) + 
                     " energy_consumption: " + str(job.energy_consumption) +
                     '\n')

def run_test(taskset_params, processor_count, scheduler_class, duration, train_rl=False, task_type='Periodic'):
    configuration = Configuration()
    configuration.duration = duration * configuration.cycles_per_ms
    configuration.state_dim = STATE_DIM

    for i, task_params in enumerate(taskset_params):
        configuration.add_task(name="T" + str(i), identifier=i, period=task_params[1],
                                task_type=task_type, list_activation_dates=[0, task_params[1], 2 * task_params[1]],
                               activation_date=0, wcet=task_params[0], deadline=task_params[1])

    for i in range(0, processor_count):
        configuration.add_processor(name="CPU " + str(i), identifier= i)

    configuration.scheduler_info.clas = scheduler_class
    configuration.scheduler_info.train = train_rl
    rl = True if scheduler_class in rl_schedulers_class else False
    configuration.scheduler_info.rl = rl
    configuration.check_all()
    configuration.save('tests/simulation_generated.xml')
    
    if not rl:
        model = Model(configuration)
        return run_single(model)
    if rl and not train_rl:
        model = Model(configuration)
        return run_single(model, load=True)

    else:
        print('Episode 0: ')
        model = Model(configuration)
        run_single(model)
        for i_episode in range(MAX_EPISODES - 2):
            print('Episode ' + str(i_episode + 1) + ':')
            model = Model(configuration)
            run_single(model, load=True)
        print('Episode ' + str(MAX_EPISODES - 1) + ':')
        model = Model(configuration)
        return run_single(model, load=True)

    # success = True
    # for task in model.results.tasks.values():
    #     if task.abort_count > 0:
    #         success = False
    #         break
    # return success

def mcm(nums):
    minimum = 1
    for i in nums:
        minimum = int(i) * int(minimum) / math.gcd(int(i), int(minimum))
    return int(minimum)

def main(argv):

    if len(argv) > 1:
        configuration = Configuration(argv[1])
        configuration.check_all()
        model = Model(configuration)
        r = run_single(model)
        export_log(r[1])

    else:
        # taskset_params_all = []
        utilizations_all = gen_randfixedsum(TASKSET_COUNT, UTILIZATION_SUM, TASKS_IN_ONE_TASKSET_COUNT)
        # periods_all = loguniform.rvs(PERIOD_MIN, PERIOD_MAX, size=TASKSET_COUNT * TASKS_IN_ONE_TASKSET_COUNT)
        periods_all = gen_periods_uniform(TASKS_IN_ONE_TASKSET_COUNT, TASKSET_COUNT, PERIOD_MIN, PERIOD_MAX, round_to_int=False)
        taskset_params_all = gen_tasksets(utilizations_all, periods_all)

        success_count = [0] * len(TESTS)
        for i, taskset_params in enumerate(taskset_params_all):
            results = [run_test(taskset_params, PROCESSOR_COUNT, test['class_name'], DURATION, train_rl=test['train']) for test in TESTS]
            for j, r in enumerate(results):
                model = r[1]
                export_log(model, str(j))
                print("Power Consumption Sum:" + str(model.energy_consumption))
                if r[0]: success_count[j] += 1
        print("Success Count: ", success_count)
        
        
if __name__ == '__main__':
    main(sys.argv)