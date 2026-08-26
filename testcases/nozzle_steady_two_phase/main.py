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

# configFile = "inputs/config_files/CM-15.3/godunov/single_phase_gas.ini"
# configFile = "inputs/config_files/CM-15.3/roe/single_phase_gas.ini"
# configFile = "inputs/config_files/CM-15.3/roe_arabi/phase_transition.ini"
# configFile = "inputs/config_files/CM-15.3/roe_arabi/single_phase_gas.ini"
# configFile = "inputs/config_files/CM-15.3/roe_arabi/two_phase.ini"
# configFile = "inputs/config_files/CM-15.3/roe_vinokur/phase_transition.ini"
# configFile = "inputs/config_files/CM-15.3/roe_vinokur/single_phase_gas.ini"
# configFile = "inputs/config_files/CM-15.3/roe_vinokur/two_phase.ini"

# configFile = "inputs/config_files/CM-10.2/test_conf.ini"
from benchmarking_tools.timing import Timer
t = Timer()
t.start()
config = Config(configFile)
driver = Driver(config=config)
driver.solve()
t.stop()
# config = Config(configFilePath = configFile)
# driver = Driver(config = config) 
# driver.solve()
# t.stop()

# config = Config(configFile)
# # driver = Driver(config, restartFilePath="Results/berana/output_B1_NX_200/step_001700.pik")
# driver = Driver(config)
# driver.solve()

# configFile = 'input.ini'
# config = Config(configFile)
# driver = Driver(config = config, restartFilePath=restart_file)
# driver.solve()



# from fluid_properties.coolprop_interface import CoolPropAbstractState_v2

# fluid = CoolPropAbstractState_v2("REFPROP", "air")
# print(fluid.PropsSI("T", "P", 100000, "D", 2))
