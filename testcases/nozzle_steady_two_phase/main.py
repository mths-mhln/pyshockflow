from pyshockflow import Driver
from pyshockflow import Config

configFile = "inputs/input_files/lettieri/L1/HOGC-PCP-SAFT.ini"
# configFile = "inputs/input_files/orchid/input_recovery.ini"


config = Config(configFile)
driver = Driver(config)
driver.solve()

# configFile = 'input.ini'
# config = Config(configFile)
# driver = Driver(config = config, restartFilePath=restart_file)
# driver.restart()