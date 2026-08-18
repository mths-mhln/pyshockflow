import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from pyshockflow.plot_styles import *

from pyshockflow.post_processing import plot_results, thermoplot_expansion_plot, unpack_simulation_results, perform_v_and_v
from pyshockflow.post_processing import HiddenPrints
from pyshockflow import Driver, Config


# files whose data to extract:
configFiles = [
    "inputs/config_files/lettieri/L1_smooth.ini",
    "inputs/config_files/lettieri/L1_friction.ini"
]

# instantiate results path list
resultPicklePaths = []

# extract their data
for configFile in configFiles:
    # Extract outputpath from config file
    config = Config(configFilePath = configFile)
    with HiddenPrints():
        driver = Driver(config = config)
    output_path = driver.resultsSubdirPath

    # Extract all pickle files stored in that output path
    pickleList = sorted(Path(f"{output_path}").glob("*.pik"))
    resultPicklePaths.append(pickleList[-1])


# Specify output variables of interest. Currently supported variables are:
# ["Density", "Pressure", "Velocity", "Mach", "Entropy", "Temperature"] 
outputVars = ["Pressure", "Mach"]  
fig = plot_results([resultPicklePaths[-1]], outputVars, showNozzleGeometry=True)
plt.show()


# plot expansion path on top of thermoplot
fig = thermoplot_expansion_plot("inputs/thermoplot/CO2.ini", resultPicklePaths, driver.config)
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
    # extract the legend keys from the filenames
    simulation_data = {}
    for i, resultPicklePath in enumerate(resultPicklePaths):
        legend_key = Path(resultPicklePath).parent.name.split(".")[0]
        simulation_data[legend_key] = unpack_simulation_results(resultPicklePath)
    comparison_results = perform_v_and_v(verification_data = v_and_v_data, simulation_data = simulation_data, show_plots = True)

    
        
        
        
    