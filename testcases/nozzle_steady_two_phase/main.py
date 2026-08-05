from pyshockflow import Driver
from pyshockflow import Config
import numpy as np


# configFile = "inputs/config_files/lettieri/L5.ini"
# configFile = "inputs/config_files/petruccelli/P1.ini"
configFile = "inputs/config_files/CM-10.0/test_conf.ini"
# configFile = "inputs/config_files/orchid/testcase.ini"
# configFile = "inputs/config_files/orchid/input_recovery.ini"


# config = Config(configFile)
# # driver = Driver(config, restartFilePath="Results/berana/output_B1_NX_200/step_001700.pik")
# driver = Driver(config)
# driver.solve()

# configFile = 'input.ini'
# config = Config(configFile)
# driver = Driver(config = config, restartFilePath=restart_file)
# driver.solve()






from fluid_properties.fluid_properties import fluid

fluid = fluid("REFPROP", "R1234ze(E)")

# extract property
print(fluid.PropsSI('T', 'D', 519.20049536, 'U', 259012.115143168))  # saturation temperature at 1 atm