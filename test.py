import sys
import time
sys.path.append("..")
from simso.core import Model
from simso.core.Env import Env
from simso.core.Scheduler import SchedulerInfo
from simso.configuration import Configuration
from PyQt5 import QtCore, QtWidgets
from simsogui.Gantt import GanttConfigure, create_gantt_window
from PyQt5.QtWidgets import QApplication
from simsogui.SimulatorWindow import SimulatorWindow
import optparse
from rl.reacher import Reacher
import rl.sac_classes as sac
import numpy as np
from gym.spaces.box import Box
from torch.utils.tensorboard import SummaryWriter
from simso.core.Criticality import Criticality
import cProfile

start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

rl_train = True
rl_test = False
scheduler_class = "simso.schedulers.EDF_VD_mono_LA_RL"
# scheduler_class = "simso.schedulers.EDF_VD_mono_LA_maxQoS"
# scheduler_class = "simso.schedulers.EDF_VD_mono_LA"
duration_ms = 80 * 1000
profile = "profile4_alpha_3_minf_0.2"
write_sim_log = True
write_speed_log = True

rl_state_length = 15

if scheduler_class == "simso.schedulers.EDF_VD_mono_LA" or scheduler_class == "simso.schedulers.EDF_VD_mono_LA_maxQoS":
    rl_train = False
    rl_test = False

max_episodes = 500
replay_buffer_size = 1e6
replay_buffer = sac.ReplayBuffer(replay_buffer_size)

action_range=1

# hyper-parameters for RL training
frame_idx   = 0
batch_size  = 300
explore_steps = 300  # for random action sampling in the beginning of training
update_itr = 1
AUTO_ENTROPY=True
DETERMINISTIC=False
hidden_dim = 128 # TODO: 修改隐藏层
rewards     = []
rl_model_path = './model/' + profile + '/sac_v2'
sim_log_path = './logs/simlog/'
speed_log_path = './logs/speedlog/'

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
    
    parser = optparse.OptionParser()
    parser.add_option('-t', '--text', help='run script instead of a GUI',
                    action='store', dest='script')
    (opts, args) = parser.parse_args()
    app = QtWidgets.QApplication(args)
    app.setOrganizationName("SimSo")
    app.setApplicationName("SimSo")
    gantt = create_gantt_window(model)
    gantt.show()
    sys.exit(app.exec_())
    
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

def main(argv):
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


        configuration.scheduler_info.rl_train = rl_train
        configuration.scheduler_info.rl_test = rl_test
        
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
        configuration.add_task(name="T6", identifier=6, period=200,
                                activation_date=0, wcet=40, deadline=200, wcet_high=40, acet=40, criticality="LO", deadline_offset=0, abort_on_miss=True)
                
        # 模拟HI任务
        # configuration.add_task(name="T7", identifier=7, period=80,
        #                        activation_date=0, wcet=15, deadline=80, wcet_high=20, acet=20, criticality="HI", deadline_offset=0, abort_on_miss=True)
        
        # Add a processor:
        configuration.add_processor(name="CPU 1", identifier=1)

        # Add a scheduler:
        configuration.scheduler_info.clas = scheduler_class

        

    configuration.check_all()
    u = 0
    hi_u = 0
    for task in configuration.task_info_list:
        u += task.wcet / task.period
        if task.criticality == Criticality.HI:
            hi_u += task.wcet_high / task.period
    print("LO Utilization:", u, "HI Utilization:", hi_u)
    
    if rl_train:
        
        writer = SummaryWriter("logs")
        
        sac_trainer = init_rl(configuration)
        configuration.scheduler_info.additional_info = {"max_deadline": max([x.deadline for x in configuration.task_info_list]), "test": False}
        
        max_reward = -10000
        for eps in range(max_episodes):
            
            configuration.scheduler_info.episode_reward = 0
            if eps < 50:
                configuration.scheduler_info.supervise_step = 0
            else:
                configuration.scheduler_info.supervise_step = 0

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
                post_train(model)
        
    if rl_test:
        
        sac_trainer = init_rl(configuration)
        sac_trainer.load_model(rl_model_path)
        configuration.scheduler_info.additional_info = {"max_deadline": max([x.deadline for x in configuration.task_info_list]), "test": True}
        configuration.scheduler_info.episode_reward = 0
        configuration.scheduler_info.supervise_step = 0
        model = Model(configuration)
        model.run_model()
        post_train(model)
        

    if not rl_train and not rl_test:
        model = Model(configuration)
        model.run_model()
        post_train(model)


if __name__ == '__main__':
    # cProfile.run('main(sys.argv)', filename="profile.out")
    main(sys.argv)