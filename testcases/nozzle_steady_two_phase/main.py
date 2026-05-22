from pyshockflow import Driver
from pyshockflow import Config

configFile = 'inputs/input_files/lettieri/input_HEOS_CoolProp_lettieri_L1.ini'
restart_file = "Results/outletPressure_136kPa_NX_200/step_002900.pik"


config = Config(configFile)
driver = Driver(config)
driver.solve()

# configFile = 'input.ini'
# config = Config(configFile)
# driver = Driver(config = config, restartFilePath=restart_file)
# driver.restart()