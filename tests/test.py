import sys
sys.path.append("..")
from simso.core import Model
from simso.configuration import Configuration


def main(argv):
    if len(argv) == 2:
        # Configuration load from a file.
        configuration = Configuration(argv[1])
    else:
        # Manual configuration:
        configuration = Configuration()

        configuration.duration = 420 * configuration.cycles_per_ms

        configuration.mc = True

        # Add tasks:
        configuration.add_task(name="T1", identifier=1, period=7,
                               activation_date=0, wcet=3, deadline=7, wcet_high=5, criticality="LO")
        configuration.add_task(name="T2", identifier=2, period=12,
                               activation_date=0, wcet=3, deadline=12, wcet_high=5, criticality="LO")
        configuration.add_task(name="T3", identifier=3, period=20,
                               activation_date=0, wcet=5, deadline=20, wcet_high=5, criticality="LO")

        # Add a processor:
        configuration.add_processor(name="CPU 1", identifier=1)

        # Add a scheduler:
        #configuration.scheduler_info.filename = "examples/RM.py"
        configuration.scheduler_info.clas = "simso.schedulers.RM"

    # Check the config before trying to run it.
    configuration.check_all()

    # Init a model from the configuration.
    model = Model(configuration)

    print(model.task_list[0].wcet_high)

    # Execute the simulation.
    # model.run_model()

    # Print logs.
    for log in model.logs:
        print(log)

main(sys.argv)