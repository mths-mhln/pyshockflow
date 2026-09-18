import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from pyshockflow.plot_styles import *

from pyshockflow.post_processing import plot_results, thermoplot_expansion_plot, unpack_simulation_results, perform_v_and_v
from pyshockflow.post_processing import HiddenPrints
from pyshockflow import Driver, Config



# files whose data to extract:
configFiles = [
    # "inputs/config_files/lettieri/L1_smooth.ini",
    # "inputs/config_files/lettieri/L1_friction.ini",
    # "inputs/config_files/lettieri/L5_smooth.ini",
    "inputs/config_files/lettieri/L5_friction.ini",
    # "inputs/config_files/petruccelli/P1.ini",
    # "inputs/config_files/petruccelli/P2.ini",
    # "inputs/config_files/petruccelli/P3.ini",
    # "inputs/config_files/petruccelli/P4.ini",
    # "inputs/config_files/berana/B1.ini",
    # "inputs/config_files/berana/B2.ini",
    # "inputs/config_files/berana/B3.ini",


    # "inputs/config_files/CM-15.3/godunov/single_phase_gas.ini",
    # "inputs/config_files/CM-15.3/roe/single_phase_gas.ini",
    # "inputs/config_files/CM-15.3/roe_arabi/phase_transition.ini",
    # "inputs/config_files/CM-15.3/roe_arabi/single_phase_gas.ini",
    # "inputs/config_files/CM-15.3/roe_arabi/two_phase.ini",
    # "inputs/config_files/CM-15.3/roe_vinokur/phase_transition.ini",
    # "inputs/config_files/CM-15.3/roe_vinokur/single_phase_gas.ini",
    # "inputs/config_files/CM-15.3/roe_vinokur/two_phase.ini",

    # "inputs/config_files/CM-10.2/test_conf.ini",
    # "inputs/config_files/CM-11.3/L1_smooth_flipped.ini",
]

# perform verification on the simulation
verificationDataFiles = [
    # "verification_data/lettieri/L1_friction__pressure.csv", 
    # "verification_data/lettieri/L1_smooth__pressure.csv"
    # "verification_data/lettieri/L5_smooth__pressure.csv"
    "verification_data/lettieri/L5_friction__pressure.csv"
    # "verification_data/petruccelli/P1__pressure.csv"
    # "verification_data/petruccelli/P2__pressure.csv"
    # "verification_data/petruccelli/P3__pressure.csv"
    # "verification_data/petruccelli/P4__pressure.csv"
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

# convert csv information to dict to comply with v_and_v function argument data format.
v_and_v_data = {}
for verificationDataPath in verificationDataFiles:
    df = pd.read_csv(verificationDataPath)
    legend_key = Path(verificationDataPath).stem
    v_and_v_data[legend_key] = {
        "meshData": {"xMeshNodes": df.iloc[1:, 0].values},
        "(final)fluidState": {"Pressure": df.iloc[1:, 1].values}
        }
# extract the legend keys from the filenames
simulation_data = {}
for resultPicklePath in resultPicklePaths:
    legend_key = Path(resultPicklePath).parent.name.split(".")[0]
    simulation_data[legend_key] = unpack_simulation_results(resultPicklePath)
comparison_results = perform_v_and_v(verification_data = v_and_v_data, simulation_data = simulation_data, show_plots = True)

    
        
        
        
    