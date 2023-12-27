import time
import rl.sac_classes as sac

use_physical_state = True

env_profile = str("3")
env_workbook_title = 'Run 35_ LKATestBenchExample'
# env_workbook_title = 'Run 3_ LKATestBenchExample'

use_CC = False

idle_speed = 0.2
alpha = 3

start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())


task_count = 20
random_task_sel = 2


rl_train = False
rl_test = True



# scheduler_class = "simso.schedulers.EDF_VD_mono_LA_RL"
# scheduler_class = "simso.schedulers.EDF_VD_mono_new"
# # scheduler_class = "simso.schedulers.EDF_VD_mono_LA_maxQoS"
# # scheduler_class = "simso.schedulers.EDF_VD_mono_LA"
# # scheduler_class = "simso.schedulers.EDF_VD_mono_LA_between_random"
# # scheduler_class = "simso.schedulers.EDF_VD_mono_LA_between_uni"
# # scheduler_class = "simso.schedulers.EDF_VD_mono_LA_between_physical"


# if scheduler_class != "simso.schedulers.EDF_VD_mono_LA_RL":
#     rl_train = False
#     rl_test = False

duration_ms = 80 * 1000 if use_physical_state else 20000
# duration_ms = 80 * 1000
profile = "profile5.1"
write_sim_log = False
write_speed_log = False
random_data_path = './randomdata/randomdata-2023-08-08-12-03-26.csv'

rl_state_length = 15 if use_physical_state else 5

max_episodes = 1000
replay_buffer_size = 1e6
replay_buffer = sac.ReplayBuffer(replay_buffer_size)

action_range=1

# hyper-parameters for RL training
frame_idx   = 0
batch_size  = 300
explore_steps = 300
# explore_steps = 400  # for random action sampling in the beginning of training
update_itr = 1
AUTO_ENTROPY=True
DETERMINISTIC=False
hidden_dim = 128 # TODO: 修改隐藏层
rewards     = []
rl_model_path = './model/' + profile + '/sac_v2'
sim_log_path = './logs/simlog/'
speed_log_path = './logs/speedlog/'

GPU = True