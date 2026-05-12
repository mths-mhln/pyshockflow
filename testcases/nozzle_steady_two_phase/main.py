from pyshockflow import Driver
from pyshockflow import Config

restart = True

if restart:
    driver = Driver(restartFilePath="Results/outletPressure_136kPa_NX_200/step_001250.pik")
    driver.restart()
else:
    configFile = 'input.ini'
    config = Config(configFile)
    driver = Driver(config)
    driver.solve()