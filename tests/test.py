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

        configuration.duration = 20 * configuration.cycles_per_ms

        configuration.mc = True

        configuration.etm = 'injectacet'

        # Add tasks:
        configuration.add_task(name="T1", identifier=1, period=10,
                               activation_date=0, wcet=1, deadline=5, wcet_high=4, criticality="HI", deadline_offset=-2, abort_on_miss=True)
        configuration.add_task(name="T2", identifier=2, period=5,
                               activation_date=0, wcet=4, deadline=10, wcet_high=7, criticality="LO", deadline_offset=0, abort_on_miss=True)
        
        # T1 WCET_LO: 1, WCET_HI: 4, ACET: 4, deadline_LO: 3, deadline_HI: 5
        # T2 WCET_LO: 4, WCET_HI: 7, ACET: 4, deadline_LO: 10, deadline_HI: 10
        # Add a processor:
        configuration.add_processor(name="CPU 1", identifier=1)

        # Add a scheduler:
        #configuration.scheduler_info.filename = "examples/RM.py"
        configuration.scheduler_info.clas = "simso.schedulers.EDF_VD_mono"

    # Check the config before trying to run it.
    configuration.check_all()

    # Init a model from the configuration.
    model = Model(configuration)

    # print(model.task_list[0].deadline_offset)

    # Execute the simulation.
    model.run_model()

    # Print logs.
    for log in model.logs:
        print(log)

main(sys.argv)