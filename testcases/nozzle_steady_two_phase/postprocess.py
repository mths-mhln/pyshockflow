import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from pyshockflow.plot_styles import *

from pyshockflow.post_processing import plot_results, thermoplot_expansion_plot, unpack_simulation_results, perform_v_and_v
from pyshockflow.post_processing import HiddenPrints
from pyshockflow import Driver, Config



# Extract outputpath from config file
config = Config(configFilePath = "inputs/config_files/CM-10.2/test_conf.ini")
with HiddenPrints():
    driver = Driver(config = config)
output_path = driver.resultsSubdirPath

# Extract all pickle files stored in that output path
pickleList = sorted(Path(f"{output_path}").glob("*.pik"))


# Specify output variables of interest. Currently supported variables are:
# ["Density", "Pressure", "Velocity", "Mach", "Entropy", "Temperature"] 
outputVars = ["Pressure", "Mach"]  
fig = plot_results([pickleList[-1]], outputVars, showNozzleGeometry=True)
plt.show()


# plot expansion path on top of thermoplot
fig = thermoplot_expansion_plot("inputs/thermoplot/thermoplot.ini", pickleList[-1], driver.config)
plt.show()


# perform verification on the simulation
verification_cases = ["lettieri/L1_pressure"]

# convert csv information to dict to comply with v_and_v function argument data format.
for verification_case in verification_cases:
    df = pd.read_csv(f"verification_data/{verification_case}.csv")
    v_and_v_data = {
        "meshData": {"xMeshNodes": df.iloc[1:, 0].values},
        "(final)fluidState": {"Pressure": df.iloc[1:, 1].values}
        }
    # extract the simulation data
    simulation_data = unpack_simulation_results(pickleList[-1])
    comparison_results = perform_v_and_v(verification_data = v_and_v_data, simulation_data = simulation_data, show_plots = True)

    
        
        
        
    