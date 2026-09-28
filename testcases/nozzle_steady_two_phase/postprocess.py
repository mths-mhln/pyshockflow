import matplotlib.pyplot as plt
import pandas as pd

from pyshockflow.plot_styles import *

from pyshockflow.post_processing import (
    VandVSpec,
    generate_fluid_state_var_profile_figs,
    generate_expansion_thermoplot,
    compute_v_and_v_metrics,
)
from pyshockflow.plotly_post_processing import (
    generate_fluid_state_var_profile_plotly, 
    generate_fluid_state_var_profile_animation_plotly,
    generate_expansion_thermoplot_plotly
)


# show fluid state variable profiles along nozzle against verification and validation data.
verificationData = VandVSpec(
    paths={
        "Pressure": (
            # "verification_data/lettieri/L1_smooth__pressure.csv",
            # "verification_data/lettieri/L1_friction__pressure.csv",
            # "verification_data/lettieri/L5_smooth__pressure.csv",
            # "verification_data/lettieri/L5_friction__pressure.csv",
            "verification_data/petruccelli/P1_friction__pressure.csv",
            # "verification_data/petruccelli/P2_friction__pressure.csv",
            # "verification_data/petruccelli/P3_friction__pressure.csv",
            # "verification_data/petruccelli/P4_friction__pressure.csv",
        ),
        "Quality": (
            # "verification_data/lettieri/L1_friction__liquid_mass_fraction.csv",
            # "verification_data/lettieri/L5_friction__liquid_mass_fraction.csv",
        ),
        "Mach": (
            # "verification_data/petruccelli/P1_friction__mach.csv",
            # "verification_data/petruccelli/P3_friction__mach.csv",
        ),
    },
    legend_labels={
        "Pressure": (
            # "L1_smooth", 
            # "L1_friction", 
            # "L5_smooth",
            # "L5_friction",
            # "P1_friction",
            "P2_verification",
            # "P3_friction",
            # "P4_friction"
        ),
    },
)

validationData = VandVSpec(
    paths={
        "Pressure": (
            # "validation_data/lettieri/L1_friction__pressure.csv",
            # "validation_data/lettieri/L5_friction__pressure.csv",
            # "validation_data/petruccelli/P1_friction__pressure.csv",
            # "validation_data/petruccelli/P2_friction__pressure.csv",
            # "validation_data/petruccelli/P3_friction__pressure.csv",
            # "validation_data/petruccelli/P4_friction__pressure.csv",
        ),
        "Quality": (
            # "validation_data/lettieri/L1_friction__liquid_mass_fraction.csv",
            # "validation_data/lettieri/L5_friction__liquid_mass_fraction.csv",
        ),
    },
    legend_labels={
        "Pressure": (
            # "L1_friction",
        ),
    },
)

fluidStateVarProfileFigs = generate_fluid_state_var_profile_figs(
    resultsFolders= [
        # "Results/lettieri/output_L1_smooth",
        # "Results/lettieri/output_L1_friction",
        # "Results/lettieri/output_L5_smooth",
        # "Results/lettieri/output_L5_friction",
        "Results/petruccelli/output_P1",
        # "Results/petruccelli/output_P2_6",
        # "Results/petruccelli/output_P3",
        # "Results/petruccelli/output_P4",
        # "Results/berana/output_B1",
        # "Results/berana/output_B2",
        # "Results/berana/output_B3",
    ],
    fluidStateVarNames=["Pressure", "Mach", "Velocity", "Density", "Temperature", "Entropy"],
    iterationIndexes=[
        -1
    ],
    simulationLegendLabels=[
        # "L1_smooth", 
        # "L1_friction", 
        # "L5_smooth",
        # "L5_friction",
        # "P1_friction",
        # "P2_friction",
        "P2 smooth",
        # "P3_friction",
        # "P4_friction",
        # "B1_friction",
        # "B2_friction",
        # "B3_friction"
    ],
    verificationData=verificationData,
    # validationData=validationData,
    showNozzleGeometry=True,
    showGhostNodes=True
)

pressureAnimation = generate_fluid_state_var_profile_animation_plotly(
    results_folder="Results/lettieri/output_L5_friction",
    fluid_state_var_name="Pressure",
    show_ghost_nodes=True,
    interval=100,
)
pressureAnimation.show()
plt.show()



# fluidStateVarProfileFigs = generate_fluid_state_var_profile_plotly(
#     results_folders= [
#         # "Results/lettieri/output_L1_smooth",
#         # "Results/lettieri/output_L1_friction",
#         # "Results/lettieri/output_L5_smooth",
#         # "Results/lettieri/output_L5_friction",
#         # "Results/petruccelli/output_P1",
#         "Results/petruccelli/output_P2",
#         # "Results/petruccelli/output_P3",
#         # "Results/petruccelli/output_P4",
#         # "Results/berana/output_B1",
#         # "Results/berana/output_B2",
#         # "Results/berana/output_B3",
#     ],
#     fluid_state_var_names=["Pressure", "Mach", "Velocity", "Density", "Temperature", "Entropy"],
#     iteration_indexes=[
#         -1
#     ],
#     simulation_legend_labels=[
#         # "L1_smooth", 
#         # "L1_friction", 
#         # "L5_smooth",
#         # "L5_friction",
#         # "P1_friction",
#         # "P2_friction",
#         "P2_smooth",
#         # "P3_friction",
#         # "P4_friction",
#         # "B1_friction",
#         # "B2_friction",
#         # "B3_friction"
#     ],
#     verification_data=verificationData,
#     validation_data=validationData,
#     show_nozzle_geometry=True,
#     show_ghost_nodes=True
# )
# for fig in fluidStateVarProfileFigs:
#     fig.show()
# # plt.show()





# compute v and v metrics for the simulation results:
v_and_v_metrics = compute_v_and_v_metrics(
    resultsFolders= [
        # "Results/lettieri/output_L1_smooth",
        # "Results/lettieri/output_L1_friction",
        # "Results/lettieri/output_L5_smooth",
        # "Results/lettieri/output_L5_friction",
        # "Results/petruccelli/output_P1",
        "Results/petruccelli/output_P2",
        # "Results/petruccelli/output_P3",
        # "Results/petruccelli/output_P4",
        # "Results/berana/output_B1",
        # "Results/berana/output_B2",
        # "Results/berana/output_B3",
    ],
    fluidStateVarNames=["Pressure", "Mach"],
    verificationData=[
        # "verification_data/lettieri/L1_smooth__pressure.csv",
        # "verification_data/lettieri/L1_friction__pressure.csv",
        # "verification_data/lettieri/L5_smooth__pressure.csv",
        # "verification_data/lettieri/L5_friction__pressure.csv",
        # "verification_data/petruccelli/P1_friction__pressure.csv",
        "verification_data/petruccelli/P2_friction__pressure.csv",
        # "verification_data/petruccelli/P3_friction__pressure.csv",
        # "verification_data/petruccelli/P4_friction__pressure.csv",
        # "verification_data/lettieri/L1_friction__liquid_mass_fraction.csv",
    ],
    # validationData=[
    #     # "validation_data/lettieri/L1_friction__pressure.csv"
    # ],
)


# # plot the expansion path on top of a thermodynamic diagram.
# fig = generate_expansion_thermoplot(
#     resultsFolders= [
#         # "Results/lettieri/output_L1_smooth",
#         # "Results/lettieri/output_L1_friction",
#         # "Results/lettieri/output_L5_smooth",
#         # "Results/lettieri/output_L5_friction",
#         # "Results/petruccelli/output_P1",
#         "Results/petruccelli/output_P2",
#         # "Results/petruccelli/output_P3",
#         # "Results/petruccelli/output_P4",
#         # "Results/berana/output_B1",
#         # "Results/berana/output_B2",
#         # "Results/berana/output_B3",
#     ],
#     iterationIndexes=[
#         -1
#     ],
#     legend_labels=[
#         # "L1_smooth", 
#         # "L1_friction", 
#         # "L5_smooth",
#         # "L5_friction",
#         # "P1_friction",
#         # "P2_friction",
#         "P2_smooth",
#         # "P3_friction",
#         # "P4_friction",
#         # "B1_friction", 
#         # "B2_friction",
#         # "B3_friction"
#     ],
#     thermoplotConfigFilePath="inputs/thermoplot/CO2.ini", 
#     showGhostNodes = True
#     )
# plt.show()


# plot the expansion path on top of a thermodynamic diagram.
fig = generate_expansion_thermoplot_plotly(
    results_folders= [
        # "Results/lettieri/output_L1_smooth",
        # "Results/lettieri/output_L1_friction",
        # "Results/lettieri/output_L5_smooth",
        # "Results/lettieri/output_L5_friction",
        # "Results/petruccelli/output_P1",
        "Results/petruccelli/output_P2",
        # "Results/petruccelli/output_P3",
        # "Results/petruccelli/output_P4",
        # "Results/berana/output_B1",
        # "Results/berana/output_B2",
        # "Results/berana/output_B3",
    ],
    iteration_indexes=[
        -1
    ],
    legend_labels=[
        # "L1_smooth", 
        # "L1_friction", 
        # "L5_smooth",
        # "L5_friction",
        # "P1_friction",
        # "P2_friction",
        # "P2_smooth",
        # "P3_friction",
        # "P4_friction",
        # "B1_friction", 
        # "B2_friction",
        # "B3_friction"
    ],
    thermoplot_config_file_path="inputs/thermoplot/CO2.ini", 
    show_ghost_nodes = True
    )
fig.show()


        
        
    