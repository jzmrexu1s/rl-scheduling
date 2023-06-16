import sys

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

rl_train = True
rl_test = False

max_episodes = 100
replay_buffer_size = 1e6
replay_buffer = sac.ReplayBuffer(replay_buffer_size)

action_range=0.5

# hyper-parameters for RL training
frame_idx   = 0
batch_size  = 300
explore_steps = 50  # for random action sampling in the beginning of training
update_itr = 1
AUTO_ENTROPY=True
DETERMINISTIC=False
hidden_dim = 512
rewards     = []
rl_model_path = './model/sac_v2'

def main(argv):
    if len(argv) == 2:
        # Configuration load from a file.
        configuration = Configuration(argv[1])
    else:
        # Manual configuration:
        configuration = Configuration()

        # ms
        configuration.duration = 888888888 * configuration.cycles_per_ms
        # configuration.duration = -1

        # configuration.mc = False

        configuration.mc = True

        # configuration.etm = 'acet'

        configuration.etm = 'injectacet'


        configuration.scheduler_info.rl_train = rl_train
        configuration.scheduler_info.rl_test = rl_test

        # Add tasks:
        configuration.add_task(name="T1", identifier=1, period=8,
                               activation_date=0, wcet=3, deadline=8, wcet_high=6, acet=6, criticality="HI", deadline_offset=0, abort_on_miss=True)
        configuration.add_task(name="T2", identifier=2, period=12,
                               activation_date=0, wcet=1, deadline=12, wcet_high=1, acet=1, criticality="LO", deadline_offset=0, abort_on_miss=True)
        configuration.add_task(name="T3", identifier=3, period=16,
                               activation_date=0, wcet=2, deadline=16, wcet_high=2, acet=2, criticality="LO", deadline_offset=0, abort_on_miss=True)
        
        # Add a processor:
        configuration.add_processor(name="CPU 1", identifier=1)

        # Add a scheduler:
        configuration.scheduler_info.clas = "simso.schedulers.EDF_VD_mono_LA_RL"

        

    configuration.check_all()

    if rl_train:
        
        action_place = Box(0.0, 1.0, [1])
        # state: current_wcet, current_ret, U, a_ego, a_lead, v_ego, v_lead
        # TODO: set state
        state_place = Box(-100, 100, [2])
        action_dim = action_place.shape[0]
        state_dim  = state_place.shape[0]

        sac_trainer=sac.SAC_Trainer(replay_buffer, hidden_dim=hidden_dim, action_range=action_range, state_dim=state_dim, action_dim=action_dim)
        
        configuration.scheduler_info.sac_trainer = sac_trainer
        configuration.scheduler_info.frame_idx = frame_idx
        configuration.scheduler_info.explore_steps = explore_steps
        configuration.scheduler_info.replay_buffer = replay_buffer
        configuration.scheduler_info.action_place = action_place
        configuration.scheduler_info.state_place = state_place

        for eps in range(max_episodes):
            
            configuration.scheduler_info.episode_reward = 0

            model = Model(configuration)
            model.run_model()
            
            if eps % 20 == 0 and eps > 0: # plot and model saving interval
                sac.plot(rewards)
                np.save('rewards', rewards)
                sac_trainer.save_model(rl_model_path)
            print('Episode: ', eps, '| Episode Reward: ', model.scheduler.episode_reward)
            rewards.append(model.scheduler.episode_reward)

        sac_trainer.save_model(rl_model_path)

        for log in model.speed_logger.range_logs:
            print(log[0], log[2])

    if rl_test:
        pass

    if not rl_train and not rl_test:
        model = Model(configuration)
        model.run_model()

        # Print logs.
        for log in model.logs:
            print(log)

        for log in model.speed_logger.range_logs:
            print(log[0], log[2])
        
        print("Power: ", model.speed_logger.default_multi_range_power(0, model.now()))
        jobs_count = 0
        aborted_jobs_count = 0
        for key in model.results.tasks.keys():
            jobs_count += len(model.results.tasks[key].jobs)
            aborted_jobs_count += model.results.tasks[key].abort_count
        print("All jobs:", jobs_count, ", aborted:", aborted_jobs_count)
        


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


main(sys.argv)