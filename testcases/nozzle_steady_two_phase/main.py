from pyshockflow import Driver
from pyshockflow import Config

restart = True

if restart:
    configFile = 'input.ini'
    config = Config(configFile)
    driver = Driver(config = config, restartFilePath="Results/outletPressure_136kPa_NX_200/step_001440.pik")
    driver.restart()
else:
    configFile = 'input.ini'
    config = Config(configFile)
    driver = Driver(config)
    driver.solve()