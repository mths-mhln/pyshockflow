import numpy as np
import matplotlib.pyplot as plt
import pickle
import os, sys

from pathlib import Path
from pyshockflow.plot_styles import *
from scipy.optimize import fsolve
from pyshockflow import Config, Driver
from pyshockflow.visualization import plotResults




### Specify the steps of the sim you want to plot
steplist = ["054600"]
# steplist = ["0000", "0250", "0500", "0750", "1000", "1250", "1260", "1280", "1300", "1320", "1340", "1360", "1380", "1400", "1420", "1440"]
# steplist = ["1440", '1441', '1442', '1443', '1444', '1445', '1446', '1447', '1448', '1449']
# steplist = ["0000", "0250", "0500", "0750", "1000", "1250", "1300", "1400", "1500", 
#             "1600", "1700", "1800", "1900", "2000", "2100", "2200", "2300", "2400", 
#             "2500", "2600", "2700", "2800", "2900", "3000", "3100", "3200", "3300", 
#             "3400", "3500", "3600", "3700", "3800", "3900", "4000", "4100", "4200", 
#             "4300", "4400", "4500", "4600", "4700", "4800", "4900", "5000", "5100", 
#             "5200", "5300", "5400", "5500", "5600", "5700", "5800", "5900", "6000", 
#             "6100", "6200", "6300", "6400", "6500", "6600", "6700", "6800", "6900", 
#             "7000", "7100", "7200", "7300", "7400", "7500", "7600", "7700", "7800", 
#             "7900", "8000", "8100", "8200", "8300", "8400", "8500", "8600", "8700", 
#             "8800", "8900", "9000", "9100", "9200", "9300"]
# steplist = ["5000", "5100", 
#             "5200", "5300", "5400", "5500", "5600", "5700", "5800", "5900", "6000", 
#             "6100", "6200", "6300", "6400", "6500", "6600", "6700", "6800", "6900", 
#             "7000", "7100", "7200", "7300", "7400", "7500", "7600", "7700", "7800", 
#             "7900", "8000", "8100", "8200", "8300", "8400", "8500", "8600", "8700", 
#             "8800", "8900", "9000", "9100", "9200", "9300"]


### mute driver print statements when initializing
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

### plot results
configFile = 'input_HEOS_CoolProp_lettieri_L1.ini'
config = Config(configFile)
with HiddenPrints():
    driver = Driver(config)
pickleList = [f"{driver.resultsPath}/step_{step}.pik" for step in steplist]

# for ideal ["X Coords","Density", "Pressure", "Velocity", "Mach", "Entropy", "TotalPressure", "Temperature", "TotalTemperature"] 
# for real ["X Coords","Density", "Pressure", "Velocity", "Mach", "Entropy", "Temperature"]
outputVars = ["Pressure", "Density"]  

plotResults(pickleList, driver, outputVars, showNozzleGeometry=True)


