import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from pyshockflow.plot_styles import *

import pyshockflow.post_processing
from pyshockflow.post_processing import results_plots, thermoplot_expansion_plot, get_expansion_data, v_and_v
from pyshockflow import Driver, Config



# Extract outputpath from config file
config = Config("inputs/config_files/lettieri/L1.ini")
with pyshockflow.post_processing.HiddenPrints():
    driver_object = Driver(config)
output_path = driver_object.resultsPath

# Extract all pickle files stored in that output path
pickleList = sorted(Path(f"{output_path}").glob("*.pik"))


# Specify output variables of interest. Currently supported variables are:
# for ideal ["X Coords","Density", "Pressure", "Velocity", "Mach", "Entropy", "TotalPressure", "Temperature", "TotalTemperature"] 
# for real ["X Coords","Density", "Pressure", "Velocity", "Mach", "Entropy", "Temperature"]
outputVars = ["Pressure", "Mach"]  
fig = results_plots([pickleList[-1]], outputVars, showNozzleGeometry=True)
plt.show()


# plot expansion path on top of thermoplot
fig = thermoplot_expansion_plot("inputs/thermoplot/thermoplot.ini", pickleList[-1], config)
plt.show()


# perform verification on the simulation
verification_cases = ["lettieri/L1_pressure"]

print(pickleList[-1])
# convert csv information to dict to comply with v_and_v function argument data format.
for verification_case in verification_cases:
    df = pd.read_csv(f"verification_data/{verification_case}.csv")
    v_and_v_data = dict.fromkeys(df.columns)
    for i, col in enumerate(df.columns):
        v_and_v_data[col] = df.iloc[1:, i].values
    
    # extract the simulation data
    simulation_data = get_expansion_data(pickleList[-1])
    comparison_results = v_and_v(verification_data = v_and_v_data, simulation_data = simulation_data, show_plots = True)

    
        
        
        
    