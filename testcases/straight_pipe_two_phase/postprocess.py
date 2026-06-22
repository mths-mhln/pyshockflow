import matplotlib.pyplot as plt

from pathlib import Path
from pyshockflow.plot_styles import *

import pyshockflow.post_processing
from pyshockflow.post_processing import results_plots, thermoplot_expansion_plot, get_expansion_data
from pyshockflow import Driver, Config



# Extrcact outputpath from config file
config = Config("inputs/input_files/CoolProp/input_REFPROP_CoolProp.ini")
with pyshockflow.post_processing.HiddenPrints():
    driver_object = Driver(config)
output_path = driver_object.resultsPath

# Extract all pickle files stored in that output path
pickleList = sorted(Path(f"{output_path}").glob("*.pik"))

# Specify output variables of interest. Currently supported variables are:
# for ideal ["X Coords","Density", "Pressure", "Velocity", "Mach", "Entropy", "TotalPressure", "Temperature", "TotalTemperature"] 
# for real ["X Coords","Density", "Pressure", "Velocity", "Mach", "Entropy", "Temperature"]
outputVars = ["Pressure", "Mach"]  

# # plot results
# fig = results_plots(pickleList, Driver, outputVars, showNozzleGeometry=False)

# # show figure
# plt.show()

# plot expansion path on top of thermoplot
fig = thermoplot_expansion_plot("inputs/thermoplot/thermoplot.ini", pickleList[-2], config)

# extract data
data = get_expansion_data(pickleList[-2])

# show figure
plt.show()

    
        
        
        
    