import matplotlib.pyplot as plt
import pandas as pd

from pyshockflow.plot_styles import *

from pyshockflow.post_processing import (
    collect_results_by_folder,
    VandVSpec,
    generate_fluid_state_var_profile_figs,
    generate_expansion_thermoplot,
    compute_v_and_v_metrics,
)


# show fluid state variable profiles along nozzle against verification and validation data.
verificationData = VandVSpec(
    paths={
        "Pressure": (
            "verification_data/lettieri/L1_smooth__pressure.csv",
            "verification_data/lettieri/L1_friction__pressure.csv",
            "verification_data/lettieri/L5_smooth__pressure.csv",
        ),
    },
    legend_labels={
        "Pressure": (
            "L1_smooth", 
            "L1_friction", 
            "L5_smooth"
        ),
    },
)

validationData = VandVSpec(
    paths={
        "Pressure": (
            "validation_data/lettieri/L1_friction__pressure.csv",
        ),
    },
    legend_labels={
        "Pressure": (
            "L1_friction",
        ),
    },
)

fluidStateVarProfileFigs = generate_fluid_state_var_profile_figs(
    resultsFolders= [
        "Results/lettieri/output_L1_smooth",
        "Results/lettieri/output_L1_friction",
        "Results/lettieri/output_L5_smooth",
        # "Results/lettieri/output_L5_friction",
        # "Results/petruccelli/output_P1",
        # "Results/petruccelli/output_P2",
        # "Results/petruccelli/output_P3",
        # "Results/petruccelli/output_P4",
        # "Results/berana/output_B1",
        # "Results/berana/output_B2",
        # "Results/berana/output_B3",
    ],
    fluidStateVarNames=["Pressure", "Mach"],
    iterationIndexes=[
        250, 
        0, 
        0
    ],
    simulationLegendLabels=[
        "L1_smooth", 
        "L1_friction", 
        "L5_smooth"
    ],
    verificationData=verificationData,
    validationData=validationData,
    showNozzleGeometry=True,
)
plt.show()

# compute v and v metrics for the simulation results:
v_and_v_metrics = compute_v_and_v_metrics(
    resultsFolders= [
        # "Results/lettieri/output_L1_smooth",
        "Results/lettieri/output_L1_friction",
        # "Results/lettieri/output_L5_smooth",
        # "Results/lettieri/output_L5_friction",
        # "Results/petruccelli/output_P1",
        # "Results/petruccelli/output_P2",
        # "Results/petruccelli/output_P3",
        # "Results/petruccelli/output_P4",
        # "Results/berana/output_B1",
        # "Results/berana/output_B2",
        # "Results/berana/output_B3",
    ],
    fluidStateVarNames=["Pressure", "Mach"],
    verificationData=[
        # "verification_data/lettieri/L1_smooth__pressure.csv",
        "verification_data/lettieri/L1_friction__pressure.csv",
        # "verification_data/lettieri/L5_smooth__pressure.csv"
    ],
    validationData=[
        "validation_data/lettieri/L1_friction__pressure.csv"
    ],
)


# plot the expansion path on top of a thermodynamic diagram.
fig = generate_expansion_thermoplot(
    resultsFolders= [
        "Results/lettieri/output_L1_smooth",
        "Results/lettieri/output_L1_friction",
        "Results/lettieri/output_L5_smooth",
        # "Results/lettieri/output_L5_friction",
        # "Results/petruccelli/output_P1",
        # "Results/petruccelli/output_P2",
        # "Results/petruccelli/output_P3",
        # "Results/petruccelli/output_P4",
        # "Results/berana/output_B1",
        # "Results/berana/output_B2",
        # "Results/berana/output_B3",
    ],
    iterationIndexes=[
        250, 
        0, 
        0
    ],
    legend_labels=[
        "L1_smooth", 
        "L1_friction", 
        "L5_smooth"
    ],
    thermoplotConfigFilePath="inputs/thermoplot/CO2.ini"
    )
plt.show()


        
        
    