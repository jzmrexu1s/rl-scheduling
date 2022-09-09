TASKSET_COUNT = 1
UTILIZATION_SUM = 1
TASKS_IN_ONE_TASKSET_COUNT = 10
DURATION = 20
PROCESSOR_COUNT = 4

PERIOD_MIN = 1
PERIOD_MAX = 10

MAX_EPISODES = 10
MAX_STEPS = 1000

STATE_DIM = 5

MODEL_DIR = 'tests/model'
LOG_DIR = 'tests/log'

TESTS = [
    {
        'class_name': 'simso.schedulers.MADDPG',
        'train': False,
    },
    {
        'class_name': 'simso.schedulers.EDF',
        'train': False,
    },

]