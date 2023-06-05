import sys

sys.path.append("..")
from simso.core import Model
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

rl_train = True
rl_test = False

max_episodes = 1000
replay_buffer_size = 1e6
replay_buffer = sac.ReplayBuffer(replay_buffer_size)

ENV = ['Reacher', 'Pendulum-v0', 'HalfCheetah-v2'][1]
if ENV == 'Reacher':
    NUM_JOINTS=2
    LINK_LENGTH=[200, 140]
    INI_JOING_ANGLES=[0.1, 0.1]
    SCREEN_SIZE=1000
    SPARSE_REWARD=False
    SCREEN_SHOT=False
    action_range = 10.0
    env=Reacher(screen_size=SCREEN_SIZE, num_joints=NUM_JOINTS, link_lengths = LINK_LENGTH, \
    ini_joint_angles=INI_JOING_ANGLES, target_pos = [369,430], render=True, change_goal=False)
    action_dim = env.num_actions
    state_dim  = env.num_observations
else:
    env = sac.NormalizedActions(gym.make(ENV))
    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    action_range=1.

# hyper-parameters for RL training
max_episodes  = 1000
if ENV ==  'Reacher':
    max_steps = 20
elif ENV ==  'Pendulum-v0':
    max_steps = 150  # Pendulum needs 150 steps per episode to learn well
elif ENV == 'HalfCheetah-v2':
    max_steps = 1000
else:
    raise NotImplementedError
frame_idx   = 0
batch_size  = 300
explore_steps = 0  # for random action sampling in the beginning of training
update_itr = 1
AUTO_ENTROPY=True
DETERMINISTIC=False
hidden_dim = 512
rewards     = []
rl_model_path = './model/sac_v2'
sac_trainer=sac.SAC_Trainer(replay_buffer, hidden_dim=hidden_dim, action_range=action_range, state_dim=state_dim, action_dim=action_dim)

def main(argv):
    if len(argv) == 2:
        # Configuration load from a file.
        configuration = Configuration(argv[1])
    else:
        # Manual configuration:
        configuration = Configuration()

        configuration.duration = 44 * configuration.cycles_per_ms

        # configuration.mc = False

        configuration.mc = True

        # configuration.etm = 'acet'

        configuration.etm = 'injectacet'


        configuration.scheduler_info.rl_train = rl_train
        configuration.scheduler_info.rl_test = rl_test

        # Add tasks:
        configuration.add_task(name="T1", identifier=1, period=8,
                               activation_date=0, wcet=2, deadline=8, wcet_high=5, acet=2, criticality="HI", deadline_offset=0, abort_on_miss=True)
        configuration.add_task(name="T2", identifier=2, period=12,
                               activation_date=0, wcet=1, deadline=12, wcet_high=1, acet=1, criticality="LO", deadline_offset=0, abort_on_miss=True)
        configuration.add_task(name="T3", identifier=3, period=16,
                               activation_date=0, wcet=2, deadline=16, wcet_high=2, acet=2, criticality="LO", deadline_offset=0, abort_on_miss=True)
        
        # Add a processor:
        configuration.add_processor(name="CPU 1", identifier=1)

        # Add a scheduler:
        #configuration.scheduler_info.filename = "examples/RM.py"
        # configuration.scheduler_info.clas = "simso.schedulers.CC_EDF"
        configuration.scheduler_info.clas = "simso.schedulers.EDF_VD_mono_CC"

        

    configuration.check_all()

    if rl_train:
        
        configuration.scheduler_info.sac_trainer = sac_trainer
        configuration.scheduler_info.env = env
        configuration.scheduler_info.frame_idx = frame_idx
        configuration.scheduler_info.explore_steps = explore_steps
        configuration.scheduler_info.replay_buffer = replay_buffer
        configuration.scheduler_info.action_dim = action_dim

        for eps in range(max_episodes):
            if ENV == 'Reacher':
                state = env.reset(SCREEN_SHOT)
            else:
                state =  env.reset()

            episode_reward = 0

            configuration.scheduler_info.state = state
            configuration.scheduler_info.episode_reward = episode_reward

            model = Model(configuration)
            model.run_model()
            
            if eps % 20 == 0 and eps > 0: # plot and model saving interval
                sac.plot(rewards)
                np.save('rewards', rewards)
                sac_trainer.save_model(rl_model_path)
            print('Episode: ', eps, '| Episode Reward: ', model.scheduler.episode_reward)
            rewards.append(model.scheduler.episode_reward)

        sac_trainer.save_model(rl_model_path)

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
        
        print(model.speed_logger.default_multi_range_power(0, model.now()))


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