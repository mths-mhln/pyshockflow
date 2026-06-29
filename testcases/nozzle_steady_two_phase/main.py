from pyshockflow import Driver
from pyshockflow import Config

configFile = "inputs/config_files/lettieri/L1.ini"
# configFile = "inputs/config_files/orchid/testcase.ini"
# configFile = "inputs/config_files/orchid/input_recovery.ini"


config = Config(configFile)
driver = Driver(config, restartFilePath="Results/berana/output_B1_NX_200/step_001700.pik")
driver.solve()

# configFile = 'input.ini'
# config = Config(configFile)
# driver = Driver(config = config, restartFilePath=restart_file)
# driver.restart()