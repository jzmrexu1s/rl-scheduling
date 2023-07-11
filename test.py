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
import gym
import numpy as np
from gym.spaces.box import Box
from torch.utils.tensorboard import SummaryWriter

start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

rl_train = False
rl_test = True
scheduler_class = "simso.schedulers.EDF_VD_mono_LA_RL"
# scheduler_class = "simso.schedulers.EDF_VD_mono_LA_maxQoS"
# scheduler_class = "simso.schedulers.EDF_VD_mono_LA"
duration_ms = 40 * 1000

if scheduler_class == "simso.schedulers.EDF_VD_mono_LA" or scheduler_class == "simso.schedulers.EDF_VD_mono_LA_maxQoS":
    rl_train = False
    rl_test = False

max_episodes = 1000
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
rl_model_path = './model/sac_v2'

def post_train(model):
    # for log in model.logs:
    #     print(log)

    # for log in model.speed_logger.range_logs:
    #     print(log[0], log[2])
    
    power = model.speed_logger.default_multi_range_power(0, model.now())
    print("Power: ", power)
    jobs_count = 0
    aborted_jobs_count = 0
    for key in model.results.tasks.keys():
        jobs_count += len(model.results.tasks[key].jobs)
        aborted_jobs_count += model.results.tasks[key].abort_count
    print("All jobs:", jobs_count, ", aborted:", aborted_jobs_count)
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
    state_place = Box(-100, 100, [10])
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
        # configuration.duration = -1

        # configuration.mc = False

        configuration.mc = True

        # configuration.etm = 'acet'

        configuration.etm = 'injectacet'


        configuration.scheduler_info.rl_train = rl_train
        configuration.scheduler_info.rl_test = rl_test

        # Add tasks:
        # configuration.add_task(name="T1", identifier=1, period=8,
        #                        activation_date=0, wcet=3, deadline=8, wcet_high=6, acet=6, criticality="HI", deadline_offset=0, abort_on_miss=True)
        # configuration.add_task(name="T2", identifier=2, period=12,
        #                        activation_date=0, wcet=1, deadline=12, wcet_high=1, acet=1, criticality="LO", deadline_offset=0, abort_on_miss=True)
        # configuration.add_task(name="T3", identifier=3, period=16,
        #                        activation_date=0, wcet=2, deadline=16, wcet_high=2, acet=2, criticality="LO", deadline_offset=0, abort_on_miss=True)
        
        configuration.add_task(name="T1", identifier=1, period=100,
                               activation_date=0, wcet=5, deadline=100, wcet_high=15, acet=5, criticality="HI", deadline_offset=0, abort_on_miss=True)
        configuration.add_task(name="T2", identifier=2, period=50,
                               activation_date=0, wcet=8, deadline=50, wcet_high=8, acet=8, criticality="LO", deadline_offset=0, abort_on_miss=True)
        configuration.add_task(name="T3", identifier=3, period=80,
                               activation_date=0, wcet=6, deadline=80, wcet_high=6, acet=6, criticality="LO", deadline_offset=0, abort_on_miss=True)
        configuration.add_task(name="T4", identifier=4, period=100,
                               activation_date=0, wcet=15, deadline=80, wcet_high=15, acet=15, criticality="LO", deadline_offset=0, abort_on_miss=True)
        
        # Add a processor:
        configuration.add_processor(name="CPU 1", identifier=1)

        # Add a scheduler:
        configuration.scheduler_info.clas = scheduler_class

        

    configuration.check_all()

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
            
            # if eps % 20 == 0 and eps > 0: # plot and model saving interval
            #     # sac.plot(rewards)
            #     np.save('rewards', rewards)
            #     sac_trainer.save_model(rl_model_path)
            if model.scheduler.episode_reward > max_reward:
                max_reward = model.scheduler.episode_reward
                sac_trainer.save_model(rl_model_path)
            
            print('Episode: ', eps, '| Episode Reward: ', model.scheduler.episode_reward)
            writer.add_scalar("Episode Reward "+ start_time, model.scheduler.episode_reward, eps)
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


main(sys.argv)