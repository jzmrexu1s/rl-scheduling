import random
import sys
import time
sys.path.append("..")
from simso.core import Model
from simso.configuration import Configuration
from PyQt5 import QtCore, QtWidgets
from simsogui.Gantt import GanttConfigure, create_gantt_window
from PyQt5.QtWidgets import QApplication
import optparse
import rl.sac_classes as sac
from gym.spaces.box import Box
from torch.utils.tensorboard import SummaryWriter
from simso.core.Criticality import Criticality
import cProfile
import pandas as pd
from config import *
from simso.generator.task_generator import gen_uunifastdiscard

def random_pick(some_list, probabilities): 
    x = random.uniform(0,1) 
    cumulative_probability = 0.0 
    for item, item_probability in zip(some_list, probabilities): 
        cumulative_probability += item_probability 
        if x < cumulative_probability:
            break 
    return item 


def post_train(model):
    # for log in model.logs:
    #     print(log)

    # for log in model.speed_logger.range_logs:
    #     print(log[0], log[2])
    
    if write_sim_log:
        with open (sim_log_path + "sim-log-" + start_time + '.log', 'w') as f:
            for log in model.logs:
                f.write(str(log[0]) + " " + log[1][0] + '\n')
    if write_speed_log:
        with open (speed_log_path + "speed-log-" + start_time + '.log', 'w') as f:
            for log in model.speed_logger.range_logs:
                f.write(str(log[0]) + " " + str(log[2]) + '\n')
    
    power = model.speed_logger.default_multi_range_power(0, model.now())
    print("Power: ", power)
    jobs_count = 0
    aborted_jobs_count = 0
    terminated_jobs_count = 0
    for key in model.results.tasks.keys():
        jobs_count += len(model.results.tasks[key].jobs)
        aborted_jobs_count += model.results.tasks[key].abort_count
        terminated_jobs_count += model.results.tasks[key].terminate_count
    print("All jobs:", jobs_count, ", aborted:", aborted_jobs_count, ", terminated:", terminated_jobs_count)
    print("Efficiency:", (jobs_count - aborted_jobs_count) / power)
    
    avg_overhead = model.scheduler.call_time / model.scheduler.call_count
    print("Overhead:", avg_overhead)
    
    parser = optparse.OptionParser()
    parser.add_option('-t', '--text', help='run script instead of a GUI',
                    action='store', dest='script')
    (opts, args) = parser.parse_args()
    # app = QtWidgets.QApplication(args)
    # app.setOrganizationName("SimSo")
    # app.setApplicationName("SimSo")
    # gantt = create_gantt_window(model)
    # gantt.show()
    # sys.exit(app.exec_())
    return jobs_count, aborted_jobs_count, terminated_jobs_count, power, (jobs_count - aborted_jobs_count) / power, avg_overhead, model.scheduler.time_series, model.scheduler.power_series, model.scheduler.terminate_series

    
def random_task_adder(configuration, df, taskset_idx):
    for i in range(task_count):
        row = df.iloc[task_count * 0 + i]
        
        configuration.add_task(name=row['name'], identifier=task_count * taskset_idx + i, period=row['period'],
                    activation_date=0, wcet=row['wcet'], deadline=row['period'], wcet_high=row['wcet_high'], acet=row['wcet'], 
                    criticality=row['criticality'], deadline_offset=0, abort_on_miss=True)
    
def init_rl(configuration):
    action_place = Box(0.0, 1.0, [1])
    # state: current_wcet, current_ret, U, a_ego, a_lead, v_ego, v_lead
    # TODO: set state
    state_place = Box(-100, 100, [rl_state_length])
    action_dim = action_place.shape[0]
    state_dim  = state_place.shape[0]

    sac_trainer=sac.SAC_Trainer(replay_buffer, hidden_dim=hidden_dim, action_range=action_range, state_dim=state_dim, action_dim=action_dim)
    
    configuration.scheduler_info.sac_trainer = sac_trainer
    configuration.scheduler_info.frame_idx = frame_idx
    configuration.scheduler_info.explore_steps = explore_steps
    configuration.scheduler_info.replay_buffer = replay_buffer
    configuration.scheduler_info.action_place = action_place
    configuration.scheduler_info.state_place = state_place
    return sac_trainer

def calculate_utilization(configuration):
    u = 0
    hi_u = 0
    for task in configuration.task_info_list:
        u += task.wcet / task.period
        if task.criticality == Criticality.HI:
            hi_u += task.wcet_high / task.period
    print("LO Utilization:", u, "HI Utilization:", hi_u)
    return u, hi_u

def main(argv, scheduler_class, rec):
    _rl_train = rl_train
    _rl_test = rl_test
    if scheduler_class != "simso.schedulers.EDF_VD_mono_LA_RL":
        _rl_train = False
        _rl_test = False
        
    if len(argv) == 2:
        # Configuration load from a file.
        configuration = Configuration(argv[1])
    else:
        # Manual configuration:
        configuration = Configuration()

        # ms
        configuration.duration = duration_ms * configuration.cycles_per_ms

        # configuration.mc = False

        configuration.mc = True

        # configuration.etm = 'acet'

        configuration.etm = 'injectacet'


        configuration.scheduler_info.rl_train = _rl_train
        configuration.scheduler_info.rl_test = _rl_test
        
        if use_physical_state:
            # 固定任务
            
            # ACC
            configuration.add_task(name="T1", identifier=1, period=100,
                                activation_date=0, wcet=12, deadline=100, wcet_high=36, acet=36, criticality="HI", deadline_offset=0, abort_on_miss=True)
            # LKA
            configuration.add_task(name="T2", identifier=2, period=100,
                                activation_date=0, wcet=9, deadline=100, wcet_high=20, acet=20, criticality="HI", deadline_offset=0, abort_on_miss=True)
            
            configuration.add_task(name="T3", identifier=3, period=80,
                                activation_date=0, wcet=6, deadline=80, wcet_high=6, acet=6, criticality="LO", deadline_offset=0, abort_on_miss=True)
            
            configuration.add_task(name="T4", identifier=4, period=100,
                                activation_date=0, wcet=15, deadline=100, wcet_high=15, acet=15, criticality="LO", deadline_offset=0, abort_on_miss=True)
            configuration.add_task(name="T5", identifier=5, period=200,
                                activation_date=0, wcet=25, deadline=200, wcet_high=25, acet=25, criticality="LO", deadline_offset=0, abort_on_miss=True)
            # configuration.add_task(name="T6", identifier=6, period=200,
            #                         activation_date=0, wcet=40, deadline=200, wcet_high=40, acet=40, criticality="LO", deadline_offset=0, abort_on_miss=True)
            # for i in range(0, rec):
            #     configuration.add_task(name="T" + str(i + 7), identifier=str(i + 7), period=100,
            #                         activation_date=0, wcet=0.01, deadline=100, wcet_high=0.01, acet=0.01, criticality="LO", deadline_offset=0, abort_on_miss=True)
                    
                    
            # 模拟HI任务
            # configuration.add_task(name="T7", identifier=7, period=80,
            #                        activation_date=0, wcet=15, deadline=80, wcet_high=20, acet=20, criticality="HI", deadline_offset=0, abort_on_miss=True)
        
        
        # Add a processor:
        configuration.add_processor(name="CPU 1", identifier=1)

        # Add a scheduler:
        configuration.scheduler_info.clas = scheduler_class

        
        calculate_utilization(configuration)
    
    
    
    if _rl_train:
        
        writer = SummaryWriter("logs")
        
        sac_trainer = init_rl(configuration)
        
        max_reward = -10000
        
        if not use_physical_state:
            df = pd.read_csv(random_data_path)
        
        for eps in range(max_episodes):
            
            
            configuration.set_task_info_list([])
            # ACC
            configuration.add_task(name="T1", identifier=1, period=100,
                                activation_date=0, wcet=12, deadline=100, wcet_high=36, acet=36, criticality="HI", deadline_offset=0, abort_on_miss=True)
            # LKA
            configuration.add_task(name="T2", identifier=2, period=100,
                                activation_date=0, wcet=9, deadline=100, wcet_high=20, acet=20, criticality="HI", deadline_offset=0, abort_on_miss=True)
            
            LO_count = random_pick([3, 4, 5, 6, 7, 8], [0.17, 0.17, 0.17, 0.17, 0.16, 0.16])
            u_all = random_pick([0.19, 0.29, 0.39, 0.49, 0.59], [0.2, 0.2, 0.2, 0.2, 0.2])
            us = gen_uunifastdiscard(1, u_all, LO_count)[0]
            for num, u in enumerate(us):
                period = random_pick([20, 25, 40, 50, 80, 100, 200, 250], [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
                configuration.add_task(name="T" + str(num + 3), identifier=num + 3, period=period,
                            activation_date=0, wcet=period * u, deadline=period, wcet_high=period * u, acet=period * u, criticality="LO", deadline_offset=0, abort_on_miss=True)
            
            if not use_physical_state:
                configuration.set_task_info_list([])
                random_task_adder(configuration, df, eps)
                
            configuration.scheduler_info.additional_info = {"max_deadline": max([x.deadline for x in configuration.task_info_list]), "test": False}
                        
            configuration.scheduler_info.episode_reward = 0
            if eps < 50:
                configuration.scheduler_info.supervise_step = 0
            else:
                configuration.scheduler_info.supervise_step = 0
            
            configuration.check_all()

            model = Model(configuration)
            model.run_model()
            
            if model.scheduler.episode_reward > max_reward:
                max_reward = model.scheduler.episode_reward
                sac_trainer.save_model(rl_model_path)
            
            print('Episode: ', eps, '| Episode Reward: ', model.scheduler.episode_reward)
            writer.add_scalar(profile + " Episode Reward " + start_time, model.scheduler.episode_reward, eps)
            rewards.append(model.scheduler.episode_reward)
            # post_train(model)
            if eps == max_episodes - 1:
                return post_train(model)
        
    if _rl_test:
        if not use_physical_state:
            df = pd.read_csv(random_data_path)
            random_task_adder(configuration, df, random_task_sel)
            calculate_utilization(configuration)
        sac_trainer = init_rl(configuration)
        sac_trainer.load_model(rl_model_path)
        configuration.scheduler_info.additional_info = {"max_deadline": max([x.deadline for x in configuration.task_info_list]), "test": True}
        configuration.scheduler_info.episode_reward = 0
        configuration.scheduler_info.supervise_step = 0
        model = Model(configuration)
        model.run_model()
        return post_train(model)
        

    if not _rl_train and not _rl_test:
        if not use_physical_state:
            df = pd.read_csv(random_data_path)
            random_task_adder(configuration, df, random_task_sel)
            calculate_utilization(configuration)
            
        model = Model(configuration)
        model.run_model()
        return post_train(model)
        


if __name__ == '__main__':
    # cProfile.run('main(sys.argv)', filename="profile.out")
    main(sys.argv, "simso.schedulers.EDF_VD_mono_LA_RL", 1)