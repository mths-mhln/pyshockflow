import matplotlib.pyplot as plt
import pandas as pd

from pyshockflow.plot_styles import *

from pyshockflow.post_processing import (
    VandVSpec,
    generate_fluid_state_var_profile_figs,
    generate_expansion_thermoplot,
    compute_v_and_v_metrics,
    render_profile_video
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
            "verification_data/lettieri/L1_smooth__pressure.csv",
            # "verification_data/lettieri/L5_smooth__pressure.csv",

            # "verification_data/lettieri/L1_friction__pressure.csv",
            # "verification_data/lettieri/L5_friction__pressure.csv",

            # "verification_data/petruccelli/P1_friction__pressure.csv",
            # "verification_data/petruccelli/P2_friction__pressure.csv",
            # "verification_data/petruccelli/P3_friction__pressure.csv",
            # "verification_data/petruccelli/P4_friction__pressure.csv",

            # "verification_data/berana/B1_friction__pressure.csv",
            # "verification_data/berana/B2_friction__pressure.csv",
            # "verification_data/berana/B3_friction__pressure.csv",
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
            "L1_smooth_verification",
            # "L5_smooth_verification",

            # "L1_friction_verification", 
            # "L5_friction_verification",

            # "P1_verification",
            # "P2_verification",
            # "P3_verification",
            # "P4_verification"

            # "B1_verification",
            # "B2_verification",
            # "B3_verification"
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
    resultsFolders=[
        "Results/lettieri/output_L1_smooth",
        # "Results/lettieri/output_L5_smooth",

        # "Results/lettieri/output_L1_friction",
        # "Results/lettieri/output_L5_friction",
        # "Results/petruccelli/output_P1",
        # "Results/petruccelli/output_P2",
        # "Results/petruccelli/output_P3",
        # "Results/petruccelli/output_P4",
        # "Results/berana/output_B1_amr",
        # "Results/berana/output_B1_200_09",
        # "Results/berana/output_B1_400_09",
        # "Results/berana/output_B1_800_09",
        # "Results/berana/output_B1_1600_09",
        # "Results/berana/output_B1",
        # "Results/berana/output_B2",
        # "Results/berana/output_B3",

        # "Results/petruccelli/output_P1_buffer",
        # "Results/petruccelli/output_P2_buffer",
        # "Results/petruccelli/output_P3_buffer",
        # "Results/petruccelli/output_P4_buffer",

        # "Results/petruccelli/output_P1_buffer_5_percent",
        # "Results/petruccelli/output_P2_buffer_5_percent",
        # "Results/petruccelli/output_P3_buffer_5_percent",
        # "Results/petruccelli/output_P4_buffer_5_percent",

        # "Results/lettieri/output_L1_smooth_5_percent",
        # "Results/lettieri/output_L5_smooth_5_percent",
        # "Results/lettieri/output_L1_friction_5_percent",
        # "Results/lettieri/output_L5_friction_5_percent",
        # "Results/petruccelli/output_P1_5_percent",
        # "Results/petruccelli/output_P2_5_percent",
        # "Results/petruccelli/output_P3_5_percent",
        # "Results/petruccelli/output_P4_5_percent",

        # "Results/lettieri/output_L1_smooth_0_percent_vel",
        # "Results/lettieri/output_L5_smooth_0_percent_vel",
        # "Results/lettieri/output_L1_friction_0_percent_vel",
        # "Results/lettieri/output_L5_friction_0_percent_vel",
        # "Results/petruccelli/output_P1_0_percent_vel",
        # "Results/petruccelli/output_P2_0_percent_vel",
        # "Results/petruccelli/output_P3_0_percent_vel",
        # "Results/petruccelli/output_P4_0_percent_vel",

        # "Results/petruccelli/output_P1_0_percent_vel_rough",
        # "Results/petruccelli/output_P2_0_percent_vel_rough",
        # "Results/petruccelli/output_P3_0_percent_vel_rough",
        # "Results/petruccelli/output_P4_0_percent_vel_rough",
    ],

    fluidStateVarNames=["Velocity", "Pressure"],

    iterationIndexes=[
        -1
    ],

    simulationLegendLabels=[
        "L1 smooth",
        # "L5 smooth",

        # "L1 friction",
        # "L5 friction",
        # "P1 friction",
        # "P2 friction",
        # "P3 friction",
        # "P4 friction",
        # "B1 friction amr",
        # "B1 friction 200",
        # "B1 friction 400",
        # "B1 friction 800",
        # "B1 friction 1600",
        # "B1 friction amr",
        # "B1 friction",
        # "B2 friction",
        # "B3 friction",

        # "L1 smooth 5 percent",
        # "L5 smooth 5 percent",
        # "L1 friction 5 percent",
        # "L5 friction 5 percent",
        # "P1 friction 5 percent",
        # "P2 friction 5 percent",
        # "P3 friction 5 percent",
        # "P4 friction 5 percent",

        # "L1 smooth 0 percent vel",
        # "L5 smooth 0 percent vel",
        # "L1 friction 0 percent vel",
        # "L5 friction 0 percent vel",
        # "P1 friction 0 percent vel",
        # "P2 friction 0 percent vel",
        # "P3 friction 0 percent vel",
        # "P4 friction 0 percent vel",

        # "P1 friction 0 percent vel rough",
        # "P2 friction 0 percent vel rough",
        # "P3 friction 0 percent vel rough",
        # "P4 friction 0 percent vel rough",
    ],

    verificationData=verificationData,
    # validationData=validationData,
    showNozzleGeometry=True,
    showGhostNodes=True, 
    separatePlots=False
)
plt.show()

# pressureAnimation = generate_fluid_state_var_profile_animation_plotly(
#     results_folder="Results/lettieri/output_L5_friction_debug_3",
#     fluid_state_var_name="Pressure",
#     show_ghost_nodes=True,
#     interval=1,
#     show_unconverged_nodes=True,
# )
# pressureAnimation.show()
# plt.show()

# render_profile_video("Results/berana/output_B1", "Pressure", "Results_berana_output_B1__pressure.mp4", speed=0.25)
# render_profile_video("Results/berana/output_B3", "Pressure", "Results_berana_output_B3__pressure.mp4", speed=0.25)
# render_profile_video("Results/berana/output_B1_amr", "Pressure", "Results_berana_output_B1__pressure.mp4", speed=0.25)



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
#         # "L1 smooth", 
#         # "L1 friction", 
#         # "L5 smooth",
#         # "L5 friction",
#         # "P1 friction",
#         # "P2 friction",
#         "P2 smooth",
#         # "P3 friction",
#         # "P4 friction",
#         # "B1 friction",
#         # "B2 friction",
#         # "B3 friction"
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
        # "Results/lettieri/output_L5_smooth",

        # "Results/lettieri/output_L1_friction",
        # "Results/lettieri/output_L5_friction",
        # "Results/petruccelli/output_P1",
        # "Results/petruccelli/output_P2",
        # "Results/petruccelli/output_P3",
        # "Results/petruccelli/output_P4",
        # "Results/berana/output_B1",
        # "Results/berana/output_B2",
        # "Results/berana/output_B3",

        # "Results/lettieri/output_L1_smooth_5_percent",
        # "Results/lettieri/output_L5_smooth_5_percent",
        # "Results/lettieri/output_L1_friction_5_percent",
        # "Results/lettieri/output_L5_friction_5_percent",
        # "Results/petruccelli/output_P1_5_percent",
        # "Results/petruccelli/output_P2_5_percent",
        # "Results/petruccelli/output_P3_5_percent",
        # "Results/petruccelli/output_P4_5_percent",

        # "Results/lettieri/output_L1_smooth_0_percent_vel",
        # "Results/lettieri/output_L5_smooth_0_percent_vel",
        # "Results/lettieri/output_L1_friction_0_percent_vel",
        # "Results/lettieri/output_L5_friction_0_percent_vel",
        # "Results/petruccelli/output_P1_0_percent_vel",
        # "Results/petruccelli/output_P2_0_percent_vel",
        # "Results/petruccelli/output_P3_0_percent_vel",
        # "Results/petruccelli/output_P4_0_percent_vel",

        # "Results/petruccelli/output_P1_0_percent_vel_rough",
        # "Results/petruccelli/output_P2_0_percent_vel_rough",
        # "Results/petruccelli/output_P3_0_percent_vel_rough",
        # "Results/petruccelli/output_P4_0_percent_vel_rough",
    ],
    fluidStateVarNames=["Pressure", "Mach"],
    verificationData=verificationData,
    # validationData=validationData
)


# plot the expansion path on top of a thermodynamic diagram.
fig = generate_expansion_thermoplot(
    resultsFolders= [
        # "Results/lettieri/output_L1_smooth",
        # "Results/lettieri/output_L5_smooth",

        # "Results/lettieri/output_L1_friction",
        # "Results/lettieri/output_L5_friction",
        # "Results/petruccelli/output_P1",
        # "Results/petruccelli/output_P2",
        # "Results/petruccelli/output_P3",
        # "Results/petruccelli/output_P4",
        # "Results/berana/output_B1",
        # "Results/berana/output_B2",
        # "Results/berana/output_B3",

        # "Results/lettieri/output_L1_smooth_5_percent",
        # "Results/lettieri/output_L5_smooth_5_percent",
        # "Results/lettieri/output_L1_friction_5_percent",
        # "Results/lettieri/output_L5_friction_5_percent",
        # "Results/petruccelli/output_P1_5_percent",
        # "Results/petruccelli/output_P2_5_percent",
        # "Results/petruccelli/output_P3_5_percent",
        # "Results/petruccelli/output_P4_5_percent",

        # "Results/lettieri/output_L1_smooth_0_percent_vel",
        # "Results/lettieri/output_L5_smooth_0_percent_vel",
        # "Results/lettieri/output_L1_friction_0_percent_vel",
        # "Results/lettieri/output_L5_friction_0_percent_vel",
        # "Results/petruccelli/output_P1_0_percent_vel",
        # "Results/petruccelli/output_P2_0_percent_vel",
        # "Results/petruccelli/output_P3_0_percent_vel",
        # "Results/petruccelli/output_P4_0_percent_vel",

        "Results/petruccelli/output_P1_0_percent_vel_rough",
        "Results/petruccelli/output_P2_0_percent_vel_rough",
        "Results/petruccelli/output_P3_0_percent_vel_rough",
        "Results/petruccelli/output_P4_0_percent_vel_rough",
    ],
    iterationIndexes=[
        # -1, -1,  # Smooth
        -1, -1, -1, -1 # Friction
        # -1, -1, -1, -1, -1, -1, -1, -1,  # 5 percent
        # -1, -1, -1, -1, -1, -1, -1, -1,  # 0 percent velocity
        # -1, -1, -1, -1,  # 0 percent velocity + rough
    ],
    simulationLegendLabels=[
        # "L1 smooth",
        # "L5 smooth",

        # "L1 friction",
        # "L5 friction",
        # "P1 friction",
        # "P2 friction",
        # "P3 friction",
        # "P4 friction",
        # "B1 friction",
        # "B2 friction",
        # "B3 friction",

        # "L1 smooth 5 percent",
        # "L5 smooth 5 percent",
        # "L1 friction 5 percent",
        # "L5 friction 5 percent",
        # "P1 friction 5 percent",
        # "P2 friction 5 percent",
        # "P3 friction 5 percent",
        # "P4 friction 5 percent",

        # "L1 smooth 0 percent vel",
        # "L5 smooth 0 percent vel",
        # "L1 friction 0 percent vel",
        # "L5 friction 0 percent vel",
        # "P1 friction 0 percent vel",
        # "P2 friction 0 percent vel",
        # "P3 friction 0 percent vel",
        # "P4 friction 0 percent vel",

        "P1 friction 0 percent vel rough",
        "P2 friction 0 percent vel rough",
        "P3 friction 0 percent vel rough",
        "P4 friction 0 percent vel rough",
    ],
    thermoplotConfigFilePath="inputs/thermoplot/CO2.ini", 
    showGhostNodes = True, 
    separatePlots = True
    )
plt.show()


# # plot the expansion path on top of a thermodynamic diagram.
# fig = generate_expansion_thermoplot_plotly(
#     results_folders= [
#         # "Results/lettieri/output_L1_smooth",
#         # "Results/lettieri/output_L1_friction",
#         # "Results/lettieri/output_L5_smooth",
#         # "Results/lettieri/output_L5_friction",
#         # "Results/petruccelli/output_P1",
#         # "Results/petruccelli/output_P2",
#         "Results/petruccelli/output_P3",
#         # "Results/petruccelli/output_P4",
#         # "Results/berana/output_B1",
#         # "Results/berana/output_B2",
#         # "Results/berana/output_B3",
#     ],
#     iteration_indexes=[
#         -1
#     ],
#     legend_labels=[
#         # "L1 smooth", 
#         # "L1 friction", 
#         # "L5 smooth",
#         # "L5 friction",
#         # "P1 friction",
#         # "P2 friction",
#         "P3 friction",
#         # "P4 friction",
#         # "B1 friction", 
#         # "B2 friction",
#         # "B3 friction"
#     ],
#     thermoplot_config_file_path="inputs/thermoplot/CO2.ini", 
#     show_ghost_nodes = True
#     )
# fig.show()


        
        
    