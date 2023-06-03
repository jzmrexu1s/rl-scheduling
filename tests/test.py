import sys
sys.path.append("..")
from simso.core import Model
from simso.configuration import Configuration
from PyQt5 import QtCore, QtWidgets
from simsogui.Gantt import GanttConfigure, create_gantt_window
from PyQt5.QtWidgets import QApplication
from simsogui.SimulatorWindow import SimulatorWindow
import optparse

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