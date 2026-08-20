from pyshockflow import Driver
from pyshockflow import Config
from benchmarking_tools.timing import Timer
t = Timer()

t.start()

configFile = "inputs/config_files/lettieri/L1_smooth.ini"
# configFile = "inputs/config_files/lettieri/L1_friction.ini"
# configFile = "inputs/config_files/lettieri/L5_smooth.ini"
# configFile = "inputs/config_files/lettieri/L5_friction.ini"
# configFile = "inputs/config_files/petruccelli/P1.ini"
# configFile = "inputs/config_files/petruccelli/P2.ini"
# configFile = "inputs/config_files/petruccelli/P3.ini"
# configFile = "inputs/config_files/petruccelli/P4.ini"
# configFile = "inputs/config_files/berana/B1.ini"
# configFile = "inputs/config_files/berana/B2.ini"
# configFile = "inputs/config_files/berana/B3.ini"

# config = Config(configFile)
# driver = Driver(config=config)
# driver.solve()
 
config = Config(configFilePath = configFile)
driver = Driver(config = config) 
driver.solve()
t.stop()

# config = Config(configFile)
# # driver = Driver(config, restartFilePath="Results/berana/output_B1_NX_200/step_001700.pik")
# driver = Driver(config)
# driver.solve()

# configFile = 'input.ini'
# config = Config(configFile)
# driver = Driver(config = config, restartFilePath=restart_file)
# driver.solve()



# from fluid_properties.coolprop_interface import CoolPropAbstractState_v2

# fluid = CoolPropAbstractState_v2("REFPROP", "CO2")
# print(fluid.PropsSI())
