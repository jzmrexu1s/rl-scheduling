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

    # app = QtWidgets.QApplication(sys.argv)
    # gantt = create_gantt_window(model)
    # ex = GanttConfigure(model, 0, 10000)
    # if ex.exec_():
    #     print(ex)

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